#!/usr/bin/env python3
"""Run batch evaluation for the frozen TCM pathology benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def _strip_user_site_from_syspath() -> None:
    """Prefer the active conda env over stale ~/.local packages."""
    user_markers = [
        "/.local/lib/python",
        str(Path.home() / ".local" / "lib"),
    ]
    sys.path[:] = [
        path for path in sys.path if not any(marker in path for marker in user_markers)
    ]


_strip_user_site_from_syspath()

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MODEL_ROOT = first_existing_path(
    REPO_ROOT / "models",
    WORKSPACE_ROOT / "models",
)
DEFAULT_BENCHMARK_ROOT = first_existing_path(
    REPO_ROOT / "data" / "benchmarks",
)
DEFAULT_CODEX_ROOT = REPO_ROOT / ".codex_tmp"

from bipharm_rag import BiPharmRAG, QueryParam
from bipharm_rag.tcm_prompt import apply_tcm_prompts
from bipharm_rag.utils import EmbeddingFunc, compute_args_hash


LATEX_BLOCK_PATTERN = re.compile(r"\$.*?\$", re.S)
NON_TEXT_PATTERN = re.compile(r"[A-Za-z0-9_`~^=+|<>/\\\\{}\\[\\]().%-]+")
PUNCT_PATTERN = re.compile(r"[，。；：、“”‘’【】（）()《》〈〉·,;:!?]")
WHITESPACE_PATTERN = re.compile(r"\s+")
SYMPTOMS_QUESTION_PATTERN = re.compile(r"患者见：(.*?)。")
SYNDROME_QUESTION_PATTERN = re.compile(
    r"在《[^》]+》这本书中，(.+?)(?:的病机|的治法|常用什么方药|若见“)",
)
CONDITION_QUESTION_PATTERN = re.compile(r"若见“(.+?)”")
FORMULA_QUESTION_PATTERN = re.compile(r"常用方“(.+?)”")
FORMULA_HERB_PATTERN = re.compile(r"含药味“(.+?)”")
ADDED_HERB_PATTERN = re.compile(r"可加“(.+?)”")
RUNTIME: "LocalRuntime | None" = None


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = text.replace("\n", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_answer(text: str) -> str:
    text = clean_text(text)
    text = LATEX_BLOCK_PATTERN.sub(" ", text)
    text = NON_TEXT_PATTERN.sub(" ", text)
    text = PUNCT_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub("", text)
    return text


def char_f1(pred: str, gold: str) -> float:
    pred = normalize_answer(pred)
    gold = normalize_answer(gold)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pred_counts: dict[str, int] = defaultdict(int)
    gold_counts: dict[str, int] = defaultdict(int)
    for ch in pred:
        pred_counts[ch] += 1
    for ch in gold:
        gold_counts[ch] += 1
    overlap = 0
    for ch, count in pred_counts.items():
        overlap += min(count, gold_counts.get(ch, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def detect_mentions(answer: str, vocab: list[str]) -> list[str]:
    answer_norm = normalize_answer(answer)
    hits: list[tuple[int, str]] = []
    for candidate in vocab:
        candidate_norm = normalize_answer(candidate)
        if candidate_norm and candidate_norm in answer_norm:
            hits.append((len(candidate_norm), candidate))
    hits.sort(key=lambda item: (-item[0], item[1]))
    deduped: list[str] = []
    seen: set[str] = set()
    for _, candidate in hits:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def contains_normalized(haystack_norm: str, needle: str) -> bool:
    needle_norm = normalize_answer(needle)
    return bool(needle_norm) and needle_norm in haystack_norm


def parse_question_symptoms(question_text: str) -> str:
    match = SYMPTOMS_QUESTION_PATTERN.search(question_text)
    return match.group(1).strip() if match else ""


def parse_question_syndrome(question_text: str) -> str:
    match = SYNDROME_QUESTION_PATTERN.search(question_text)
    return match.group(1).strip() if match else ""


def parse_question_condition(question_text: str) -> str:
    match = CONDITION_QUESTION_PATTERN.search(question_text)
    return match.group(1).strip() if match else ""


def parse_question_formula(question_text: str) -> str:
    match = FORMULA_QUESTION_PATTERN.search(question_text)
    return match.group(1).strip() if match else ""


def parse_question_herb(question_text: str) -> str:
    for pattern in (FORMULA_HERB_PATTERN, ADDED_HERB_PATTERN):
        match = pattern.search(question_text)
        if match:
            return match.group(1).strip()
    return ""


def dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        key = normalize_answer(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def question_syndrome(question: dict[str, Any]) -> str:
    return str(question.get("syndrome") or parse_question_syndrome(question["question"])).strip()


def question_formula(question: dict[str, Any]) -> str:
    return str(question.get("formula_name") or parse_question_formula(question["question"])).strip()


def question_condition(question: dict[str, Any]) -> str:
    return str(question.get("condition") or parse_question_condition(question["question"])).strip()


def is_hidden_herb_reverse_question(question: dict[str, Any]) -> bool:
    return question.get("question_type") in {"herb_from_formula_clue", "herb_from_condition_clue"}


def question_herb_terms(question: dict[str, Any], *, include_structured_fields: bool = True) -> list[str]:
    values = [parse_question_herb(question["question"])]
    if include_structured_fields:
        values.extend(
            [
                str(question.get("herb_name", "")).strip(),
                str(question.get("matched_pharm_herb", "")).strip(),
            ]
        )
    return dedupe_preserve_order(values)


def is_dual_bridge_question(question: dict[str, Any]) -> bool:
    return question["question_type"] in {"herb_efficacy_from_case", "herb_indications_from_condition"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class CaseRouter:
    """Structured case memory used to calibrate answers from source-book cases."""

    def __init__(
        self,
        cases: list[dict[str, Any]],
        *,
        use_typed_case_memory: bool = True,
        enable_modification_routing: bool = True,
    ) -> None:
        self.cases = cases
        self.use_typed_case_memory = use_typed_case_memory
        self.enable_modification_routing = enable_modification_routing
        self.cases_by_symptoms: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.cases_by_syndrome: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.cases_by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            self.cases_by_symptoms[normalize_answer(case.get("symptoms", ""))].append(case)
            self.cases_by_syndrome[normalize_answer(case.get("syndrome", ""))].append(case)
            for formula_name in self.case_formula_aliases(case):
                self.cases_by_formula[normalize_answer(formula_name)].append(case)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        use_typed_case_memory: bool = True,
        enable_modification_routing: bool = True,
    ) -> "CaseRouter":
        return cls(
            load_jsonl(path),
            use_typed_case_memory=use_typed_case_memory,
            enable_modification_routing=enable_modification_routing,
        )

    def case_formula_aliases(self, case: dict[str, Any]) -> list[str]:
        return dedupe_preserve_order(
            [str(case.get("formula_name", "")).strip(), *[str(item).strip() for item in case.get("formula_aliases", [])]]
        )

    def case_matches_formula(self, case: dict[str, Any], formula_name: str) -> bool:
        formula_norm = normalize_answer(formula_name)
        if not formula_norm:
            return False
        return any(normalize_answer(alias) == formula_norm for alias in self.case_formula_aliases(case))

    def case_contains_herb_terms(self, case: dict[str, Any], herb_terms: list[str]) -> bool:
        if not herb_terms:
            return False
        formula_text = normalize_answer(str(case.get("formula_text", "")))
        raw_excerpt = normalize_answer(str(case.get("raw_text_excerpt", "")))
        if any(term and normalize_answer(term) in formula_text for term in herb_terms):
            return True
        return any(term and normalize_answer(term) in raw_excerpt for term in herb_terms)

    def modification_contains_herb_terms(self, modification: dict[str, Any] | None, herb_terms: list[str]) -> bool:
        if modification is None or not herb_terms:
            return False
        herb_norms = {normalize_answer(str(item)) for item in modification.get("herbs", []) if str(item).strip()}
        for term in herb_terms:
            term_norm = normalize_answer(term)
            if term_norm and term_norm in herb_norms:
                return True
        return False

    def best_modification(
        self,
        case: dict[str, Any],
        condition_text: str,
    ) -> tuple[dict[str, Any] | None, bool, float]:
        if not self.enable_modification_routing:
            return None, False, 0.0
        condition_norm = normalize_answer(condition_text)
        exact_match = None
        best_match = None
        best_score = 0.0
        for mod in case.get("modification_pairs", []):
            mod_condition = str(mod.get("condition", ""))
            mod_norm = normalize_answer(mod_condition)
            if condition_norm and mod_norm == condition_norm:
                exact_match = mod
                break
            score = char_f1(condition_text, mod_condition)
            if score > best_score:
                best_score = score
                best_match = mod
        if exact_match is not None:
            return exact_match, True, 1.0
        return best_match, False, best_score

    def score_case_candidate(
        self,
        case: dict[str, Any],
        question: dict[str, Any],
        context: str | None,
        condition_text: str = "",
    ) -> tuple[float, dict[str, Any]]:
        context = context or ""
        context_norm = normalize_answer(context)
        question_type = question["question_type"]
        syndrome_text = question_syndrome(question)
        formula_text_query = question_formula(question)
        herb_terms = question_herb_terms(
            question,
            include_structured_fields=not is_hidden_herb_reverse_question(question),
        )
        score = 0.0

        def add_contains(text: str, weight: float) -> float:
            return weight if contains_normalized(context_norm, text) else 0.0

        pathogenesis = str(case.get("pathogenesis", ""))
        therapy = str(case.get("therapy", ""))
        formula_name = str(case.get("formula_name", ""))
        symptoms = str(case.get("symptoms", ""))
        raw_excerpt = str(case.get("raw_text_excerpt", ""))
        formula_text = str(case.get("formula_text", ""))

        if syndrome_text:
            score += 20.0 if normalize_answer(case.get("syndrome", "")) == normalize_answer(syndrome_text) else char_f1(
                syndrome_text,
                str(case.get("syndrome", "")),
            ) * 8.0
        if formula_text_query:
            if self.case_matches_formula(case, formula_text_query):
                score += 24.0
            else:
                score += max((char_f1(formula_text_query, alias) for alias in self.case_formula_aliases(case)), default=0.0) * 10.0

        score += add_contains(pathogenesis, 5.0)
        score += add_contains(therapy, 5.0)
        score += add_contains(formula_name, 6.0)
        score += add_contains(case.get("syndrome", ""), 2.0)
        score += char_f1(context, raw_excerpt) * 8.0
        score += char_f1(context, formula_text) * 3.0
        score += char_f1(context, symptoms) * 2.0

        for alias in case.get("formula_aliases", []):
            score += add_contains(str(alias), 3.0)

        if question_type == "pathogenesis_from_syndrome":
            score += char_f1(context, pathogenesis) * 6.0
        elif question_type == "therapy_from_syndrome":
            score += char_f1(context, therapy) * 6.0
        elif question_type == "formula_from_syndrome":
            score += char_f1(context, formula_name) * 8.0
        elif question_type == "modification_from_condition" and self.enable_modification_routing:
            mod, exact_match, mod_score = self.best_modification(case, condition_text)
            if exact_match:
                score += 50.0
            score += mod_score * 20.0
            if mod is not None:
                score += add_contains(mod.get("condition", ""), 4.0)
                score += char_f1(context, str(mod.get("raw_clause", ""))) * 10.0
                for herb in mod.get("herbs", []):
                    score += add_contains(str(herb), 2.0)
        elif question_type == "herb_efficacy_from_case":
            if self.case_contains_herb_terms(case, herb_terms):
                score += 30.0
            for herb in herb_terms:
                score += add_contains(herb, 4.0)
        elif question_type == "herb_indications_from_condition" and self.enable_modification_routing:
            mod, exact_match, mod_score = self.best_modification(case, condition_text)
            if exact_match:
                score += 50.0
            score += mod_score * 20.0
            if mod is not None:
                if self.modification_contains_herb_terms(mod, herb_terms):
                    score += 30.0
                score += add_contains(mod.get("condition", ""), 4.0)
                score += char_f1(context, str(mod.get("raw_clause", ""))) * 10.0

        debug = {
            "case_id": case.get("case_id"),
            "syndrome": case.get("syndrome"),
            "pathogenesis": pathogenesis,
            "therapy": therapy,
            "formula_name": formula_name,
            "herb_terms": herb_terms,
            "score": round(score, 4),
        }
        return score, debug

    def generic_case_text(self, case: dict[str, Any]) -> str:
        parts = [
            str(case.get("symptoms", "")).strip(),
            str(case.get("syndrome", "")).strip(),
            str(case.get("pathogenesis", "")).strip(),
            str(case.get("therapy", "")).strip(),
            str(case.get("formula_name", "")).strip(),
            str(case.get("formula_text", "")).strip(),
            str(case.get("raw_text_excerpt", "")).strip(),
        ]
        for mod in case.get("modification_pairs", []):
            parts.append(str(mod.get("condition", "")).strip())
            parts.append(str(mod.get("raw_clause", "")).strip())
            parts.extend(str(item).strip() for item in mod.get("herbs", []))
        return " ".join(part for part in parts if part)

    def score_case_candidate_generic(
        self,
        case: dict[str, Any],
        question: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        query_text = str(question.get("question", "")).strip()
        score = 0.0
        score += char_f1(query_text, str(case.get("raw_text_excerpt", ""))) * 18.0
        score += char_f1(query_text, str(case.get("formula_text", ""))) * 12.0
        score += char_f1(query_text, str(case.get("symptoms", ""))) * 10.0
        score += char_f1(query_text, str(case.get("syndrome", ""))) * 8.0
        score += char_f1(query_text, str(case.get("pathogenesis", ""))) * 6.0
        score += char_f1(query_text, str(case.get("therapy", ""))) * 6.0
        score += char_f1(query_text, str(case.get("formula_name", ""))) * 8.0

        question_norm = normalize_answer(query_text)
        case_text_norm = normalize_answer(self.generic_case_text(case))
        if question_norm and question_norm in case_text_norm:
            score += 12.0

        formula_name = question_formula(question)
        if formula_name and contains_normalized(case_text_norm, formula_name):
            score += 6.0

        syndrome = question_syndrome(question)
        if syndrome and contains_normalized(case_text_norm, syndrome):
            score += 6.0

        herb_terms = question_herb_terms(
            question,
            include_structured_fields=not is_hidden_herb_reverse_question(question),
        )
        for herb in herb_terms:
            if contains_normalized(case_text_norm, herb):
                score += 3.0

        debug = {
            "case_id": case.get("case_id"),
            "syndrome": case.get("syndrome"),
            "formula_name": case.get("formula_name"),
            "score": round(score, 4),
            "routing_variant": "generic_text_rerank",
        }
        return score, debug

    def answer_from_case(self, question: dict[str, Any], case: dict[str, Any]) -> str:
        question_type = question["question_type"]
        if question_type == "syndrome_from_symptoms":
            return str(case.get("syndrome", "")).strip()
        if question_type == "pathogenesis_from_syndrome":
            return str(case.get("pathogenesis", "")).strip()
        if question_type == "therapy_from_syndrome":
            return str(case.get("therapy", "")).strip()
        if question_type == "formula_from_syndrome":
            return str(case.get("formula_name", "")).strip()
        if question_type == "chain_from_symptoms":
            return (
                f"证型：{str(case.get('syndrome', '')).strip()}；"
                f"病机：{str(case.get('pathogenesis', '')).strip()}；"
                f"治法：{str(case.get('therapy', '')).strip()}；"
                f"方药：{str(case.get('formula_name', '')).strip()}"
            )
        if question_type == "modification_from_condition":
            mod, _, _ = self.best_modification(case, parse_question_condition(question["question"]))
            if mod is None:
                return ""
            herbs = [str(item).strip() for item in mod.get("herbs", []) if str(item).strip()]
            return "、".join(herbs)
        return ""

    def route_case(
        self,
        question: dict[str, Any],
        context: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        case, debug = self.select_case(question, context=context)
        if case is None:
            return "", debug
        return self.answer_from_case(question, case), debug

    def select_case(
        self,
        question: dict[str, Any],
        context: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        question_type = question["question_type"]
        debug: dict[str, Any] = {
            "question_type": question_type,
            "selected_case_id": None,
            "selected_syndrome": None,
            "selection_mode": None,
            "num_candidates": 0,
            "candidate_scores": [],
            "use_typed_case_memory": self.use_typed_case_memory,
            "enable_modification_routing": self.enable_modification_routing,
        }

        if not self.use_typed_case_memory:
            scored = [
                (score, case, case_debug)
                for case in self.cases
                for score, case_debug in [self.score_case_candidate_generic(case, question)]
            ]
            scored.sort(key=lambda item: (item[0], str(item[1].get("case_id", ""))), reverse=True)
            if not scored:
                debug["selection_mode"] = "no_candidate"
                return None, debug
            best_score, best_case, _ = scored[0]
            debug["selection_mode"] = "generic_case_rerank"
            debug["num_candidates"] = len(scored)
            debug["selected_case_id"] = best_case.get("case_id")
            debug["selected_syndrome"] = best_case.get("syndrome")
            debug["selected_score"] = round(best_score, 4)
            debug["candidate_scores"] = [item[2] for item in scored[:3]]
            return best_case, debug

        candidates: list[dict[str, Any]]
        if question_type in {"syndrome_from_symptoms", "chain_from_symptoms"}:
            symptoms = parse_question_symptoms(question["question"])
            debug["question_symptoms"] = symptoms
            candidates = self.cases_by_symptoms.get(normalize_answer(symptoms), [])
            if len(candidates) == 1:
                case = candidates[0]
                debug["selection_mode"] = "symptoms_exact"
                debug["num_candidates"] = 1
                debug["selected_case_id"] = case.get("case_id")
                debug["selected_syndrome"] = case.get("syndrome")
                return case, debug
        else:
            syndrome = question_syndrome(question)
            debug["question_syndrome"] = syndrome
            candidates = self.cases_by_syndrome.get(normalize_answer(syndrome), [])
            if not candidates:
                formula_name = question_formula(question)
                if formula_name:
                    debug["question_formula"] = formula_name
                    candidates = self.cases_by_formula.get(normalize_answer(formula_name), [])

        formula_name = question_formula(question)
        if formula_name:
            debug["question_formula"] = formula_name
            exact_formula_cases = [case for case in candidates if self.case_matches_formula(case, formula_name)]
            if exact_formula_cases:
                candidates = exact_formula_cases

        herb_terms = question_herb_terms(
            question,
            include_structured_fields=not is_hidden_herb_reverse_question(question),
        )
        if herb_terms:
            debug["question_herbs"] = herb_terms
        if question_type == "herb_efficacy_from_case":
            herb_cases = [case for case in candidates if self.case_contains_herb_terms(case, herb_terms)]
            if herb_cases:
                candidates = herb_cases

        debug["num_candidates"] = len(candidates)
        if not candidates:
            debug["selection_mode"] = "no_candidate"
            return None, debug

        if self.enable_modification_routing and question_type in {"modification_from_condition", "herb_indications_from_condition"}:
            condition_text = question_condition(question)
            debug["question_condition"] = condition_text
            exact_condition_cases = []
            for case in candidates:
                mod, exact_match, _ = self.best_modification(case, condition_text)
                if exact_match:
                    if question_type == "herb_indications_from_condition" and not self.modification_contains_herb_terms(mod, herb_terms):
                        continue
                    exact_condition_cases.append(case)
            if len(exact_condition_cases) == 1:
                case = exact_condition_cases[0]
                debug["selection_mode"] = "condition_exact"
                debug["num_candidates"] = len(exact_condition_cases)
                debug["selected_case_id"] = case.get("case_id")
                debug["selected_syndrome"] = case.get("syndrome")
                return case, debug
            if exact_condition_cases:
                candidates = exact_condition_cases
                debug["num_candidates"] = len(candidates)
            elif question_type == "herb_indications_from_condition":
                herb_condition_cases = []
                for case in candidates:
                    mod, _, _ = self.best_modification(case, condition_text)
                    if self.modification_contains_herb_terms(mod, herb_terms):
                        herb_condition_cases.append(case)
                if herb_condition_cases:
                    candidates = herb_condition_cases
                    debug["num_candidates"] = len(candidates)

        if len(candidates) == 1:
            case = candidates[0]
            debug["selection_mode"] = "syndrome_unique"
            debug["selected_case_id"] = case.get("case_id")
            debug["selected_syndrome"] = case.get("syndrome")
            return case, debug

        condition_text = question_condition(question) if question_type in {"modification_from_condition", "herb_indications_from_condition"} else ""
        scored = [
            (score, case, case_debug)
            for case in candidates
            for score, case_debug in [self.score_case_candidate(case, question, context, condition_text=condition_text)]
        ]
        scored.sort(key=lambda item: (item[0], str(item[1].get("case_id", ""))), reverse=True)
        best_score, best_case, _ = scored[0]
        debug["selection_mode"] = "context_rerank"
        debug["selected_case_id"] = best_case.get("case_id")
        debug["selected_syndrome"] = best_case.get("syndrome")
        debug["selected_score"] = round(best_score, 4)
        debug["candidate_scores"] = [item[2] for item in scored[:3]]
        return best_case, debug


class LocalRuntime:
    """Local Qwen + BGE runtime reused across all queries in one process."""

    def __init__(
        self,
        llm_dir: Path,
        emb_dir: Path,
        emb_device: str,
        llm_device_map: str,
        llm_dtype: str,
        llm_max_token_size: int,
        emb_max_length: int,
        emb_batch_size: int,
        llm_backend: str = "local",
        codex_model: str = "gpt-5.4",
        codex_reasoning_effort: str = "low",
        codex_workdir: Path | None = None,
        codex_audit_dir: Path | None = None,
        codex_strict_mode: bool = False,
        codex_bin: str = "codex",
        load_embedding: bool = True,
    ) -> None:
        self.llm_dir = llm_dir
        self.emb_dir = emb_dir
        self.emb_device = emb_device
        self.llm_max_token_size = llm_max_token_size
        self.emb_max_length = emb_max_length
        self.emb_batch_size = emb_batch_size
        self.llm_backend = llm_backend
        self.codex_model = codex_model
        self.codex_reasoning_effort = codex_reasoning_effort
        self.codex_workdir = codex_workdir or (REPO_ROOT.parent / "_tmp_codex_llm")
        self.codex_audit_dir = codex_audit_dir or (self.codex_workdir / "audit")
        self.codex_strict_mode = codex_strict_mode
        self.codex_bin = codex_bin
        self.load_embedding = load_embedding

        if self.llm_backend == "local":
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=resolve_torch_dtype(llm_dtype) if torch.cuda.is_available() else torch.float32,
                device_map=llm_device_map,
            )
            self.llm_model.eval()
        elif self.llm_backend == "codex-exec":
            self.llm_tokenizer = None
            self.llm_model = None
            self.codex_workdir.mkdir(parents=True, exist_ok=True)
            self.codex_audit_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unsupported llm backend: {self.llm_backend}")

        self.emb_tokenizer = None
        self.emb_model = None
        self.emb_dim = 1024
        if load_embedding:
            self.emb_tokenizer = AutoTokenizer.from_pretrained(
                emb_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.emb_model = AutoModel.from_pretrained(
                emb_dir,
                dtype=torch.float16 if emb_device.startswith("cuda") else torch.float32,
                trust_remote_code=True,
                local_files_only=True,
                weights_only=False,
            )
            self.emb_model.to(emb_device)
            self.emb_model.eval()
            self.emb_dim = getattr(self.emb_model.config, "hidden_size", 1024)

    def llm_input_device(self) -> torch.device:
        if self.llm_model is None:
            raise RuntimeError("Local LLM backend is not initialized.")
        return next(self.llm_model.parameters()).device

    def build_messages(self, prompt: str, system_prompt: str | None, history_messages: list[dict] | None) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        return messages

    def build_chat_prompt(self, messages: list[dict]) -> str:
        if self.llm_backend != "local" or self.llm_tokenizer is None:
            raise RuntimeError("Chat prompt templating is only available for the local backend.")
        if hasattr(self.llm_tokenizer, "apply_chat_template"):
            try:
                return self.llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        lines = []
        for msg in messages:
            lines.append(f"{msg['role']}: {msg['content']}")
        lines.append("assistant:")
        return "\n".join(lines)

    def build_codex_prompt(self, messages: list[dict], max_new_tokens: int) -> str:
        lines = [
            "You are a plain text completion backend inside an automated benchmark.",
            "The working directory is intentionally empty. Any tool or shell usage invalidates the run.",
            "Never execute commands. Never inspect files. Never browse. Never ask follow-up questions.",
            "Use only the conversation text below and return only the assistant response content for the final user request.",
            "If the instructions require strict JSON, output strict JSON only and nothing else.",
            f"Keep the answer concise and within about {max_new_tokens} generated tokens.",
            "",
            "Conversation:",
        ]
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "")).strip()
            lines.append(f"[{role}]")
            lines.append(content)
            lines.append("")
        lines.append("[ASSISTANT]")
        return "\n".join(lines)

    @staticmethod
    def parse_codex_json_stream(raw_stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
        events: list[dict[str, Any]] = []
        non_json_lines: list[str] = []
        for raw_line in raw_stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                non_json_lines.append(line)
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                non_json_lines.append(line)
        return events, non_json_lines

    @staticmethod
    def extract_tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_events: list[dict[str, Any]] = []
        for event in events:
            if event.get("type") not in {"item.started", "item.completed"}:
                continue
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type and item_type != "agent_message":
                tool_events.append(event)
        return tool_events

    @staticmethod
    def extract_last_agent_message(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            if event.get("type") != "item.completed":
                continue
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                return str(item.get("text", "")).strip()
        return ""

    def generate_via_codex_exec(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        messages = self.build_messages(prompt, system_prompt, history_messages)
        codex_prompt = self.build_codex_prompt(messages, max_new_tokens=max_new_tokens)
        audit_key = compute_args_hash(
            self.llm_backend,
            self.codex_model,
            self.codex_reasoning_effort,
            messages,
            max_new_tokens,
        )
        audit_file = self.codex_audit_dir / f"{int(time.time() * 1000)}_{audit_key}.json"

        with tempfile.TemporaryDirectory(dir=self.codex_workdir, prefix="codex_exec_run_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            output_path = temp_dir_path / "last_message.txt"
            cmd = [
                self.codex_bin,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-C",
                str(temp_dir_path),
                "-s",
                "read-only",
                "-m",
                self.codex_model,
                "-c",
                f"model_reasoning_effort={json.dumps(self.codex_reasoning_effort)}",
                "-c",
                'model_verbosity="low"',
                "--color",
                "never",
                "--json",
                "-o",
                str(output_path),
                "-",
            ]
            completed = subprocess.run(
                cmd,
                input=codex_prompt,
                text=True,
                capture_output=True,
                check=False,
            )

            events, non_json_lines = self.parse_codex_json_stream(completed.stdout)
            tool_events = self.extract_tool_events(events)
            response = ""
            if output_path.exists():
                response = output_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not response:
                response = self.extract_last_agent_message(events)

            audit_payload = {
                "timestamp": int(time.time()),
                "llm_backend": self.llm_backend,
                "codex_model": self.codex_model,
                "codex_reasoning_effort": self.codex_reasoning_effort,
                "codex_strict_mode": self.codex_strict_mode,
                "messages": messages,
                "codex_prompt": codex_prompt,
                "temp_dir": str(temp_dir_path),
                "command": cmd,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "non_json_lines": non_json_lines,
                "tool_events": tool_events,
                "response": response,
            }
            audit_file.write_text(
                json.dumps(audit_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if completed.returncode != 0:
            raise RuntimeError(
                "codex exec failed with exit code "
                f"{completed.returncode}; audit={audit_file}"
            )
        if not response:
            raise RuntimeError(f"codex exec returned an empty response; audit={audit_file}")
        if self.codex_strict_mode:
            if completed.stderr.strip():
                raise RuntimeError(f"codex exec produced stderr output; audit={audit_file}")
            if non_json_lines:
                raise RuntimeError(f"codex exec produced non-JSON stream output; audit={audit_file}")
            if tool_events:
                raise RuntimeError(f"codex exec attempted tool usage; audit={audit_file}")
        return response

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        if self.llm_backend == "codex-exec":
            return self.generate_via_codex_exec(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                max_new_tokens=max_new_tokens,
            )
        messages = self.build_messages(prompt, system_prompt, history_messages)
        text = self.build_chat_prompt(messages)
        inputs = self.llm_tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.llm_input_device()) for key, value in inputs.items()}
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )
        # GLM-4 remote code is not compatible with the new DynamicCache path in our transformers build.
        if getattr(getattr(self.llm_model, "config", None), "model_type", "") == "chatglm":
            generate_kwargs["use_cache"] = False

        with torch.inference_mode():
            outputs = self.llm_model.generate(
                **inputs,
                **generate_kwargs,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def local_llm_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        max_new_tokens = int(kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256)))
        hashing_kv = kwargs.get("hashing_kv")
        messages = self.build_messages(prompt, system_prompt, history_messages)
        cache_key = None
        if hashing_kv is not None:
            cache_key = compute_args_hash(
                self.llm_backend,
                self.codex_model if self.llm_backend == "codex-exec" else str(self.llm_dir),
                messages,
                max_new_tokens,
            )
            cached = await hashing_kv.get_by_id(cache_key)
            if cached is not None:
                return str(cached.get("return", ""))

        response = self.generate(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            max_new_tokens=max_new_tokens,
        )
        if hashing_kv is not None and cache_key is not None:
            await hashing_kv.upsert(
                {
                    cache_key: {
                        "return": response,
                        "model": self.codex_model if self.llm_backend == "codex-exec" else str(self.llm_dir),
                    }
                }
            )
        return response

    async def local_embedding_func(self, texts: list[str]) -> np.ndarray:
        if not self.load_embedding or self.emb_tokenizer is None or self.emb_model is None:
            raise RuntimeError("Embedding model is not loaded for this runtime.")
        all_embeddings = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.emb_batch_size):
                batch = texts[start:start + self.emb_batch_size]
                encoded = self.emb_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.emb_max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.emb_device) for key, value in encoded.items()}
                outputs = self.emb_model(**encoded)
                last_hidden_state = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                embeddings = mean_pool(last_hidden_state, encoded["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.float().cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)


def require_runtime() -> "LocalRuntime":
    if RUNTIME is None:
        raise RuntimeError("Local runtime is not initialized.")
    return RUNTIME


async def local_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs: Any,
) -> str:
    return await require_runtime().local_llm_func(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def local_embedding_func(texts: list[str]) -> np.ndarray:
    return await require_runtime().local_embedding_func(texts)


def load_input_text(path: Path) -> list[str]:
    if path.suffix.lower() != ".jsonl":
        return [path.read_text(encoding="utf-8")]

    records: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            section_path = payload.get("section_path") or []
            if isinstance(section_path, list) and section_path:
                title = " > ".join(str(part) for part in section_path if part)
                records.append(f"{title}\n{text}")
            else:
                records.append(text)
    return records


def maybe_insert_data(rag: BiPharmRAG, data_file: Path | None, working_dir: Path) -> None:
    kv_file = working_dir / "kv_store_full_docs.json"
    if kv_file.exists() and kv_file.stat().st_size > 10:
        return
    if data_file is None:
        raise ValueError(
            f"working_dir is empty: {working_dir}. "
            "Please pass --data-file to build the cache, or reuse a populated working_dir."
        )
    texts = load_input_text(data_file)
    rag.insert(texts)


def strip_reasoning_artifacts(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"<think>\s*", "", text, flags=re.I)
    text = re.sub(r"\s*</think>", "", text, flags=re.I)
    text = text.replace("```json", "```").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text.strip()


FAIL_RESPONSE_PATTERNS = (
    "sorry, i'm not able to provide an answer to that question",
    "i do not have enough information",
    "not enough information",
    "cannot answer from the provided context",
    "无法根据提供的上下文回答",
    "没有足够信息回答",
    "信息不足，无法回答",
)

REASONING_MARKERS = (
    "thinking process",
    "analyze the request",
    "scan the context",
    "scan the knowledge base",
    "step-by-step instruction",
)

FINAL_ANSWER_MARKERS = (
    r"(?i)\bfinal answer\b\s*[:：]",
    r"(?i)\banswer\b\s*[:：]",
    r"最终答案\s*[:：]",
    r"最终回答\s*[:：]",
    r"答案\s*[:：]",
    r"回答\s*[:：]",
)


def strip_reference_section(text: str) -> str:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^(?:#{1,6}\s*)?(references|参考文献|参考资料)\s*[:：]?\s*$", stripped, flags=re.I):
            break
        if re.match(r"^[-*]\s*\[\d+\]", stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def try_parse_jsonish_dict(text: str) -> dict[str, Any] | None:
    candidate = strip_reasoning_artifacts(text)
    if not candidate:
        return None

    variants: list[str] = [candidate]
    balanced = extract_balanced_json_object(candidate)
    if balanced:
        variants.append(balanced)
    if not candidate.startswith("{") and re.search(r'["\']?[A-Za-z_][A-Za-z0-9_]*["\']?\s*[:：=]', candidate):
        variants.append("{" + candidate + "}")

    seen: set[str] = set()
    for variant in variants:
        normalized = variant.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        json_like = normalized.replace("：", ":")
        json_like = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*[:=])', r'\1"\2":', json_like)
        json_like = re.sub(r",(\s*[}\]])", r"\1", json_like)
        json_like = re.sub(r"=\s*", ": ", json_like)

        for parser_input, parser in (
            (normalized, json.loads),
            (json_like, json.loads),
            (normalized, ast.literal_eval),
            (json_like, ast.literal_eval),
        ):
            try:
                payload = parser(parser_input)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(payload, dict):
                return payload
    return None


def extract_json_block(text: str) -> dict[str, Any] | None:
    return try_parse_jsonish_dict(text)


def question_schema(question_type: str) -> tuple[dict[str, Any], str]:
    if question_type == "syndrome_from_symptoms":
        return {"syndrome": ""}, "字段 `syndrome` 填证型名称。"
    if question_type == "pathogenesis_from_syndrome":
        return {"pathogenesis": ""}, "字段 `pathogenesis` 填病机原文短语。"
    if question_type == "therapy_from_syndrome":
        return {"therapy": ""}, "字段 `therapy` 填治法原文短语。"
    if question_type == "formula_from_syndrome":
        return {"formula": ""}, "字段 `formula` 填方剂名称。"
    if question_type == "chain_from_symptoms":
        return {
            "syndrome": "",
            "pathogenesis": "",
            "therapy": "",
            "formula": "",
        }, "四个字段都尽量直接复制原文短语。"
    if question_type == "modification_from_condition":
        return {"herbs": []}, "字段 `herbs` 填加减项，保留药名或方名数组；只输出明确出现的项。"
    if question_type == "herb_efficacy_from_case":
        return {"efficacy": ""}, "字段 `efficacy` 填药典中的功效原文短语。"
    if question_type == "herb_indications_from_condition":
        return {"indications": ""}, "字段 `indications` 填药典中的主治/适应证原文短语。"
    return {"answer": ""}, "字段 `answer` 填最短直接答案。"


def render_structured_answer(question_type: str, payload: dict[str, Any]) -> str:
    def text_value(key: str) -> str:
        value = payload.get(key, "")
        if isinstance(value, list):
            return "、".join(str(item).strip() for item in value if str(item).strip())
        if value is None:
            return ""
        return str(value).strip()

    if question_type == "syndrome_from_symptoms":
        return text_value("syndrome")
    if question_type == "pathogenesis_from_syndrome":
        return text_value("pathogenesis")
    if question_type == "therapy_from_syndrome":
        return text_value("therapy")
    if question_type == "formula_from_syndrome":
        return text_value("formula")
    if question_type == "chain_from_symptoms":
        return (
            f"证型：{text_value('syndrome')}；"
            f"病机：{text_value('pathogenesis')}；"
            f"治法：{text_value('therapy')}；"
            f"方药：{text_value('formula')}"
        )
    if question_type == "modification_from_condition":
        return text_value("herbs")
    if question_type == "herb_efficacy_from_case":
        return text_value("efficacy")
    if question_type == "herb_indications_from_condition":
        return text_value("indications")
    return text_value("answer")


def normalize_generated_answer(text: str, question_type: str | None = None) -> str:
    candidate = strip_reasoning_artifacts(text)
    if not candidate:
        return ""

    candidate = candidate.replace("[no-context]", "").strip()
    candidate = strip_reference_section(candidate)
    if not candidate:
        return ""

    payload = extract_json_block(candidate)
    if payload is not None and question_type:
        rendered = render_structured_answer(question_type, payload).strip()
        if rendered:
            return rendered

    lowered = candidate.lower()
    if any(pattern in lowered for pattern in FAIL_RESPONSE_PATTERNS):
        return ""

    for marker_pattern in FINAL_ANSWER_MARKERS:
        matches = list(re.finditer(marker_pattern, candidate))
        if matches:
            candidate = candidate[matches[-1].end() :].strip()
            break

    candidate = strip_reference_section(candidate)
    if not candidate:
        return ""

    if any(marker in candidate.lower() for marker in REASONING_MARKERS):
        return ""

    cleaned_lines: list[str] = []
    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if re.match(r"^(?:#{1,6}\s*)?(references|参考文献|参考资料)\b", line, flags=re.I):
            break
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line)
        line = re.sub(r"^(?:#{1,6}\s*)?(?:answer|回答|答案|最终答案)\s*[:：]\s*", "", line, flags=re.I)
        if line:
            cleaned_lines.append(line)

    normalized = "\n".join(cleaned_lines).strip()
    if not normalized:
        return ""
    if any(marker in normalized.lower() for marker in REASONING_MARKERS):
        return ""
    return normalized


def build_structured_prompt(question: dict[str, Any], context: str) -> tuple[str, str]:
    schema, rule = question_schema(question["question_type"])
    system_prompt = (
        "你是中医基准评测抽取器。"
        "只能依据给定上下文抽取答案。"
        "禁止解释、禁止扩写、禁止同义改写、禁止补充上下文中不存在的信息。"
        "如果无法确定，请把对应字段置为空字符串，数组字段置为空数组。"
        "输出必须是严格 JSON，不要 markdown，不要额外说明，不要输出思考过程。"
        "最终回复必须从字符 { 开始，并以 } 结束。"
    )
    user_prompt = (
        f"题型：{question['question_type']}\n"
        f"问题：{question['question']}\n"
        f"输出 JSON Schema：{json.dumps(schema, ensure_ascii=False)}\n"
        f"规则：{rule}\n"
        "请优先从上下文里的原文短语直接复制，尤其是“证候/病机/治法/方药/加减”这些字段。\n"
        "上下文如下：\n"
        f"{context}"
    )
    return system_prompt, user_prompt


def best_bridge_context_lines(raw_context: str, anchors: list[str], keywords: list[str], max_lines: int = 24) -> list[str]:
    scored_lines: list[tuple[float, int, str]] = []
    for idx, line in enumerate(raw_context.splitlines()):
        text = line.strip()
        if not text or text.startswith("-----") or text == "```":
            continue
        line_norm = normalize_answer(text)
        if not line_norm:
            continue
        score = 0.0
        for anchor in anchors:
            if contains_normalized(line_norm, anchor):
                score += 6.0
        for keyword in keywords:
            if keyword in text or contains_normalized(line_norm, keyword):
                score += 2.0
        if score > 0:
            scored_lines.append((score, idx, text))
    if not scored_lines:
        return []

    top_rows = sorted(scored_lines, key=lambda item: (item[0], -item[1]), reverse=True)[:max_lines]
    selected_indices = {idx for _, idx, _ in top_rows}
    deduped: list[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(raw_context.splitlines()):
        if idx not in selected_indices:
            continue
        text = line.strip()
        key = normalize_answer(text)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def render_casebridge_memory(question: dict[str, Any], case: dict[str, Any] | None, case_router: CaseRouter | None) -> str:
    lines: list[str] = []
    syndrome = question_syndrome(question) or (str(case.get("syndrome", "")).strip() if case else "")
    formula_name = question_formula(question) or (str(case.get("formula_name", "")).strip() if case else "")
    condition_text = question_condition(question)
    herb_terms = question_herb_terms(question)

    if syndrome:
        lines.append(f"证型: {syndrome}")
    if formula_name:
        lines.append(f"方剂: {formula_name}")
    if herb_terms:
        lines.append(f"目标药味: {' / '.join(herb_terms)}")
    if condition_text:
        lines.append(f"加减条件: {condition_text}")

    if case is not None:
        if case.get("pathogenesis"):
            lines.append(f"病例病机: {str(case.get('pathogenesis', '')).strip()}")
        if case.get("therapy"):
            lines.append(f"病例治法: {str(case.get('therapy', '')).strip()}")
        if case.get("formula_text"):
            lines.append(f"病例方药组成: {str(case.get('formula_text', '')).strip()}")
        if condition_text and case_router is not None:
            modification, _, _ = case_router.best_modification(case, condition_text)
            if modification is not None:
                if modification.get("condition"):
                    lines.append(f"匹配加减条件: {str(modification.get('condition', '')).strip()}")
                herbs = [str(item).strip() for item in modification.get("herbs", []) if str(item).strip()]
                if herbs:
                    lines.append(f"匹配加减项: {'、'.join(herbs)}")
                if modification.get("raw_clause"):
                    lines.append(f"匹配加减原文: {str(modification.get('raw_clause', '')).strip()}")
    return "\n".join(lines)


def build_casebridge_context(
    question: dict[str, Any],
    raw_context: str,
    case: dict[str, Any] | None,
    case_router: CaseRouter | None,
) -> str:
    herb_terms = question_herb_terms(question)
    anchors = dedupe_preserve_order(
        [
            question_syndrome(question),
            question_formula(question),
            question_condition(question),
            *herb_terms,
            str(case.get("syndrome", "")).strip() if case else "",
            str(case.get("formula_name", "")).strip() if case else "",
        ]
    )
    keywords = ["药材", "药味", "功效", "主治", "常用于", "适应证", "功能主治", "方药", "加减"]
    if question["question_type"] == "herb_efficacy_from_case":
        keywords.extend(["消食", "止呕", "化痰", "解毒"])
    if question["question_type"] == "herb_indications_from_condition":
        keywords.extend(["主治", "用于", "便秘", "腹痛", "黄疸"])

    filtered_lines = best_bridge_context_lines(raw_context, anchors=anchors, keywords=keywords)
    memory_text = render_casebridge_memory(question, case, case_router)
    if filtered_lines:
        return f"【病例结构化记忆】\n{memory_text}\n\n【药典检索过滤片段】\n" + "\n".join(filtered_lines)
    return f"【病例结构化记忆】\n{memory_text}\n\n【药典原始检索上下文】\n{raw_context[:6000]}"


def build_casebridge_prompt(question: dict[str, Any], context: str) -> tuple[str, str]:
    schema, rule = question_schema(question["question_type"])
    herb_terms = " / ".join(question_herb_terms(question))
    if question["question_type"] == "herb_efficacy_from_case":
        extra_rule = "只抽取目标药味对应的功效，不要输出主治、证型、方剂或其他药味的信息。"
    else:
        extra_rule = "只抽取目标药味对应的主治/常用于信息，不要输出功效、证型、方剂或其他药味的信息。"
    system_prompt = (
        "你是中医病例-药典桥接抽取器。"
        "必须先依据病例锚点确认目标证型、方剂、加减条件与目标药味，再只从药典相关片段抽取答案。"
        "禁止解释、禁止扩写、禁止同义改写、禁止混入其他药味的信息。"
        "如果无法确定，请把对应字段置为空字符串，数组字段置为空数组。"
        "输出必须是严格 JSON，不要 markdown，不要额外说明，不要输出思考过程。"
        "最终回复必须从字符 { 开始，并以 } 结束。"
    )
    user_prompt = (
        f"题型：{question['question_type']}\n"
        f"问题：{question['question']}\n"
        f"目标药味：{herb_terms}\n"
        f"输出 JSON Schema：{json.dumps(schema, ensure_ascii=False)}\n"
        f"规则：{rule}\n"
        f"额外要求：{extra_rule}\n"
        "如果检索片段里出现多个药味，只保留与目标药味完全对应的一项。\n"
        "上下文如下：\n"
        f"{context}"
    )
    return system_prompt, user_prompt


def clean_bridge_candidate(text: str) -> str:
    text = text.replace("<SEP>", " ")
    text = text.replace("、", "，")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\d+$", "", text)
    text = text.strip(" \t\n\r,，;；。:：")
    return text


def trim_bridge_answer(question_type: str, text: str) -> str:
    candidate = clean_bridge_candidate(text)
    if not candidate:
        return ""

    candidate = re.sub(r"^(?:功效|主治|功能主治|适应证|常用于|用于)\s*[:：]\s*", "", candidate)

    stop_markers = [
        "主治",
        "功能主治",
        "适应证",
        "用法与用量",
        "禁忌",
        "注意",
        "注意事项",
        "不良反应",
        "规格",
        "贮藏",
        "来源",
        "疗程",
    ]
    if question_type == "herb_indications_from_condition":
        stop_markers = [
            "功效",
            "用法与用量",
            "禁忌",
            "注意",
            "注意事项",
            "不良反应",
            "规格",
            "贮藏",
            "来源",
            "疗程",
        ]

    stop_pattern = r"(?:^|[。；;，,]\s*)(?:" + "|".join(re.escape(marker) for marker in stop_markers) + r")\s*[:：]"
    match = re.search(stop_pattern, candidate)
    if match:
        candidate = candidate[: match.start()]

    candidate = re.split(r"(?:。|；|;)\s*(?:另|并|且)?(?:可见|可用于|用于|主治|功效|用法与用量|禁忌|来源|疗程)\s*[:：]?", candidate, maxsplit=1)[0]
    candidate = clean_bridge_candidate(candidate)
    return candidate


def bridge_candidate_is_confident(question_type: str, candidate: str) -> bool:
    candidate_norm = normalize_answer(candidate)
    if len(candidate_norm) < 6:
        return False
    bad_markers = [
        "适用人群",
        "临床场景",
        "直接决定",
        "治疗主轴",
        "核心实体",
        "总体而言",
        "该药材",
    ]
    if any(marker in candidate for marker in bad_markers):
        return False
    segments = [normalize_answer(part) for part in re.split(r"[，,]", candidate) if normalize_answer(part)]
    if question_type == "herb_efficacy_from_case" and len(segments) >= 3 and all(len(part) <= 2 for part in segments):
        return False
    return True


def heuristic_casebridge_answer(question: dict[str, Any], context: str) -> str:
    herb_terms = question_herb_terms(question)
    if not herb_terms:
        return ""

    candidates: list[tuple[float, str]] = []
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_norm = normalize_answer(line)
        if not any(term and normalize_answer(term) in line_norm for term in herb_terms):
            continue

        extracted: list[tuple[float, str]] = []
        if question["question_type"] == "herb_efficacy_from_case":
            for match in re.finditer(r"功效[:：]\s*([^<]+?)(?:<SEP>|$)", line):
                extracted.append((120.0, trim_bridge_answer(question["question_type"], match.group(1))))
        else:
            for match in re.finditer(r"主治[:：]\s*([^<]+?)(?:<SEP>|$)", line):
                extracted.append((120.0, trim_bridge_answer(question["question_type"], match.group(1))))

        for base_score, candidate in extracted:
            if not candidate or not bridge_candidate_is_confident(question["question_type"], candidate):
                continue
            score = base_score + len(normalize_answer(candidate)) * 0.2
            if "功效:" in line or "主治:" in line:
                score += 15.0
            if any(term == candidate for term in herb_terms):
                score -= 50.0
            candidates.append((score, candidate))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], len(normalize_answer(item[1]))), reverse=True)
    best = candidates[0][1]
    return best


def casebridge_answer_from_context(
    runtime: LocalRuntime,
    question: dict[str, Any],
    context: str,
    max_new_tokens: int,
) -> tuple[str, str | None]:
    heuristic_answer = heuristic_casebridge_answer(question, context)
    if heuristic_answer:
        return heuristic_answer, json.dumps({"heuristic_answer": heuristic_answer}, ensure_ascii=False)

    system_prompt, user_prompt = build_casebridge_prompt(question, context)
    raw_response = runtime.generate(
        user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )
    payload = extract_json_block(raw_response)
    if payload is None:
        fallback_system = (
            "你是中医病例-药典桥接短答案抽取器。"
            "只能保留目标药味对应的一条最短答案，不要解释。"
            "如果不确定，就只输出空字符串。"
        )
        fallback_prompt = (
            f"题型：{question['question_type']}\n"
            f"问题：{question['question']}\n"
            f"目标药味：{' / '.join(question_herb_terms(question))}\n"
            "请直接输出最终答案，不要解释。\n"
            f"上下文：\n{context}"
        )
        fallback_answer = runtime.generate(
            fallback_prompt,
            system_prompt=fallback_system,
            max_new_tokens=max_new_tokens,
        )
        return trim_bridge_answer(question["question_type"], fallback_answer), raw_response
    return trim_bridge_answer(question["question_type"], render_structured_answer(question["question_type"], payload)), raw_response


def structured_answer_from_context(
    runtime: LocalRuntime,
    question: dict[str, Any],
    context: str,
    max_new_tokens: int,
) -> tuple[str, str | None]:
    system_prompt, user_prompt = build_structured_prompt(question, context)
    raw_response = runtime.generate(
        user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )
    payload = extract_json_block(raw_response)
    if payload is None:
        fallback_system = (
            "你是中医短答案抽取器。"
            "只能从上下文复制原文，不要解释。"
            "如果不确定，就只输出空字符串。"
        )
        fallback_prompt = (
            f"题型：{question['question_type']}\n"
            f"问题：{question['question']}\n"
            "请直接输出最终答案，不要解释。\n"
            f"上下文：\n{context}"
        )
        return runtime.generate(
            fallback_prompt,
            system_prompt=fallback_system,
            max_new_tokens=max_new_tokens,
        ), raw_response
    return render_structured_answer(question["question_type"], payload), raw_response


def direct_answer_without_retrieval(
    runtime: LocalRuntime,
    question_text: str,
    response_type: str,
    max_new_tokens: int,
) -> str:
    system_prompt = (
        "你是中医问答基线模型。"
        "现在不能访问任何外部检索上下文。"
        "请仅依据模型自身知识直接回答。"
        "如果把握不足，请明确说无法确定，不要伪造出处。"
    )
    user_prompt = (
        f"问题：{question_text}\n"
        f"回答要求：{response_type}\n"
        "请直接给出最终答案，不要解释推理过程。"
    )
    return runtime.generate(
        user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )


def score_question(
    question: dict[str, Any],
    answer: str,
    vocab: dict[str, list[str]],
    context: str | None = None,
) -> dict[str, Any]:
    aliases = question.get("gold_answer_aliases") or [question["gold_answer"]]
    answer_norm = normalize_answer(answer)
    alias_norms = [normalize_answer(alias) for alias in aliases if alias]
    target_hit = int(any(alias and alias in answer_norm for alias in alias_norms))
    exact_match = int(any(alias and alias == answer_norm for alias in alias_norms))
    best_char_f1 = max((char_f1(answer, alias) for alias in aliases), default=0.0)

    slot_hits = 0
    slot_expected = 0
    slot_predicted = 0
    per_slot: dict[str, Any] = {}

    for slot_name, gold_value in question["gold_slots"].items():
        gold_values = gold_value if isinstance(gold_value, list) else [gold_value]
        gold_values = [str(value) for value in gold_values if value]
        predicted_values = detect_mentions(answer, vocab.get(slot_name, []))
        gold_norms = {normalize_answer(value) for value in gold_values}
        pred_norms = {normalize_answer(value) for value in predicted_values}
        hits = len(gold_norms & pred_norms)
        slot_hits += hits
        slot_expected += len(gold_norms)
        slot_predicted += len(pred_norms)
        precision = hits / len(pred_norms) if pred_norms else 0.0
        recall = hits / len(gold_norms) if gold_norms else 1.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_slot[slot_name] = {
            "gold": gold_values,
            "predicted": predicted_values,
            "hits": hits,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    slot_precision = slot_hits / slot_predicted if slot_predicted else 0.0
    slot_recall = slot_hits / slot_expected if slot_expected else 1.0
    slot_f1 = 0.0 if slot_precision + slot_recall == 0 else 2 * slot_precision * slot_recall / (slot_precision + slot_recall)

    context_hit = None
    if context is not None:
        context_norm = normalize_answer(context)
        context_hit = int(any(alias and alias in context_norm for alias in alias_norms))

    return {
        "target_hit": target_hit,
        "exact_match": exact_match,
        "char_f1": best_char_f1,
        "slot_hit_count": slot_hits,
        "slot_expected_count": slot_expected,
        "slot_predicted_count": slot_predicted,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
        "all_slots_hit": int(slot_expected > 0 and slot_hits == slot_expected),
        "context_hit": context_hit,
        "per_slot": per_slot,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_questions": 0,
            "target_hit_rate": 0.0,
            "exact_match_rate": 0.0,
            "avg_char_f1": 0.0,
            "avg_slot_precision": 0.0,
            "avg_slot_recall": 0.0,
            "avg_slot_f1": 0.0,
            "all_slots_hit_rate": 0.0,
        }
    num_rows = len(rows)
    metrics = {
        "num_questions": num_rows,
        "target_hit_rate": sum(row["metrics"]["target_hit"] for row in rows) / num_rows,
        "exact_match_rate": sum(row["metrics"]["exact_match"] for row in rows) / num_rows,
        "avg_char_f1": sum(row["metrics"]["char_f1"] for row in rows) / num_rows,
        "avg_slot_precision": sum(row["metrics"]["slot_precision"] for row in rows) / num_rows,
        "avg_slot_recall": sum(row["metrics"]["slot_recall"] for row in rows) / num_rows,
        "avg_slot_f1": sum(row["metrics"]["slot_f1"] for row in rows) / num_rows,
        "all_slots_hit_rate": sum(row["metrics"]["all_slots_hit"] for row in rows) / num_rows,
    }
    context_rows = [row for row in rows if row["metrics"]["context_hit"] is not None]
    if context_rows:
        metrics["context_hit_rate"] = sum(row["metrics"]["context_hit"] for row in context_rows) / len(context_rows)
    return metrics


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    type_names = sorted({row["question_type"] for row in rows})
    family_names = sorted({row["question_family"] for row in rows})
    for name in type_names:
        subset = [row for row in rows if row["question_type"] == name]
        by_type[name] = aggregate_rows(subset)
    for name in family_names:
        subset = [row for row in rows if row["question_family"] == name]
        by_family[name] = aggregate_rows(subset)
    return {
        "overall": aggregate_rows(rows),
        "by_question_type": by_type,
        "by_question_family": by_family,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_summary_markdown(path: Path, mode: str, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        f"# Eval Summary: {mode}",
        "",
        "## Overall",
        "",
        f"- num_questions: {overall['num_questions']}",
        f"- target_hit_rate: {overall['target_hit_rate']:.4f}",
        f"- exact_match_rate: {overall['exact_match_rate']:.4f}",
        f"- avg_char_f1: {overall['avg_char_f1']:.4f}",
        f"- avg_slot_precision: {overall['avg_slot_precision']:.4f}",
        f"- avg_slot_recall: {overall['avg_slot_recall']:.4f}",
        f"- avg_slot_f1: {overall['avg_slot_f1']:.4f}",
        f"- all_slots_hit_rate: {overall['all_slots_hit_rate']:.4f}",
    ]
    if "context_hit_rate" in overall:
        lines.append(f"- context_hit_rate: {overall['context_hit_rate']:.4f}")

    lines.extend(
        [
            "",
            "## By Question Type",
            "",
            "| Type | N | Hit | Exact | CharF1 | SlotRecall | AllSlots |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in summary["by_question_type"].items():
        lines.append(
            f"| {name} | {metrics['num_questions']} | {metrics['target_hit_rate']:.4f} | "
            f"{metrics['exact_match_rate']:.4f} | {metrics['avg_char_f1']:.4f} | "
            f"{metrics['avg_slot_recall']:.4f} | {metrics['all_slots_hit_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## By Question Family",
            "",
            "| Family | N | Hit | Exact | CharF1 | SlotRecall | AllSlots |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, metrics in summary["by_question_family"].items():
        lines.append(
            f"| {name} | {metrics['num_questions']} | {metrics['target_hit_rate']:.4f} | "
            f"{metrics['exact_match_rate']:.4f} | {metrics['avg_char_f1']:.4f} | "
            f"{metrics['avg_slot_recall']:.4f} | {metrics['all_slots_hit_rate']:.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_examples_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    interesting_types = [
        "chain_from_symptoms",
        "modification_from_condition",
        "syndrome_from_symptoms",
        "herb_efficacy_from_case",
        "herb_indications_from_condition",
    ]
    lines = ["# Qualitative Examples", ""]
    for question_type in interesting_types:
        subset = [row for row in rows if row["question_type"] == question_type]
        if not subset:
            continue
        subset.sort(key=lambda row: (row["metrics"]["all_slots_hit"], row["metrics"]["char_f1"]), reverse=True)
        best = subset[:2]
        worst = sorted(subset, key=lambda row: (row["metrics"]["all_slots_hit"], row["metrics"]["char_f1"]))[:2]
        lines.append(f"## {question_type}")
        lines.append("")
        for label, samples in [("Best", best), ("Worst", worst)]:
            lines.append(f"### {label}")
            lines.append("")
            for row in samples:
                lines.append(f"- Question ID: {row['question_id']}")
                lines.append(f"- Question: {row['question']}")
                lines.append(f"- Gold: {row['gold_answer']}")
                lines.append(f"- Answer: {row['answer']}")
                lines.append(
                    f"- Metrics: hit={row['metrics']['target_hit']}, "
                    f"char_f1={row['metrics']['char_f1']:.4f}, "
                    f"slot_recall={row['metrics']['slot_recall']:.4f}"
                )
                if row.get("context"):
                    lines.append(f"- Context: {clean_text(row['context'])[:800]}")
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_overall_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    include_context = any("context_hit_rate" in summary["overall"] for summary in summaries.values())
    fieldnames = [
        "mode",
        "num_questions",
        "target_hit_rate",
        "exact_match_rate",
        "avg_char_f1",
        "avg_slot_precision",
        "avg_slot_recall",
        "avg_slot_f1",
        "all_slots_hit_rate",
    ]
    if include_context:
        fieldnames.append("context_hit_rate")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for mode, summary in summaries.items():
            row = {"mode": mode}
            row.update(summary["overall"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "piwei_v1" / "qa_benchmark.jsonl",
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "piwei_v1" / "vocab.json",
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["cog", "naive", "cog-hybrid", "cog-entity", "cog-theme"],
    )
    parser.add_argument(
        "--question-types",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--question-families",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--collect-context",
        action="store_true",
    )
    parser.add_argument(
        "--answer-strategy",
        choices=["direct", "structured", "case-router", "casebridge"],
        default="direct",
    )
    parser.add_argument(
        "--response-type",
        type=str,
        default="请用紧凑中文作答，优先给出原文短语或按题目要求列点，不要展开。",
    )
    parser.add_argument(
        "--structured-max-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--llm-dir",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "Qwen" / "Qwen3.5-2B",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["local", "codex-exec"],
        default="local",
    )
    parser.add_argument(
        "--codex-model",
        type=str,
        default="gpt-5.4",
    )
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="low",
    )
    parser.add_argument(
        "--codex-workdir",
        type=Path,
        default=DEFAULT_CODEX_ROOT,
    )
    parser.add_argument(
        "--codex-audit-dir",
        type=Path,
        default=DEFAULT_CODEX_ROOT / "audit",
    )
    parser.add_argument(
        "--codex-strict-mode",
        action="store_true",
        help="Enable benchmark-only hard checks for codex-exec JSON stream, stderr, and tool usage.",
    )
    parser.add_argument(
        "--emb-dir",
        type=Path,
        default=DEFAULT_MODEL_ROOT / "BAAI" / "bge-m3",
    )
    parser.add_argument(
        "--emb-device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--llm-device-map",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--llm-dtype",
        type=str,
        default="float16",
    )
    parser.add_argument(
        "--llm-max-token-size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--emb-max-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--emb-batch-size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    apply_tcm_prompts()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = load_jsonl(args.benchmark_file)
    if args.question_types:
        benchmark = [row for row in benchmark if row["question_type"] in set(args.question_types)]
    if args.question_families:
        benchmark = [row for row in benchmark if row["question_family"] in set(args.question_families)]
    if args.max_questions > 0:
        benchmark = benchmark[: args.max_questions]
    vocab = json.loads(args.vocab_file.read_text(encoding="utf-8"))
    if args.case_file is None:
        default_case_file = args.benchmark_file.parent / "cases.jsonl"
        args.case_file = default_case_file if default_case_file.exists() else None
    case_router = (
        CaseRouter.load(args.case_file)
        if args.answer_strategy in {"case-router", "casebridge"} and args.case_file
        else None
    )
    use_rag = any(mode != "llm-only" for mode in args.modes)

    global RUNTIME
    runtime = LocalRuntime(
        llm_dir=args.llm_dir,
        emb_dir=args.emb_dir,
        emb_device=args.emb_device,
        llm_device_map=args.llm_device_map,
        llm_dtype=args.llm_dtype,
        llm_max_token_size=args.llm_max_token_size,
        emb_max_length=args.emb_max_length,
        emb_batch_size=args.emb_batch_size,
        llm_backend=args.llm_backend,
        codex_model=args.codex_model,
        codex_reasoning_effort=args.codex_reasoning_effort,
        codex_workdir=args.codex_workdir,
        codex_audit_dir=args.codex_audit_dir,
        codex_strict_mode=args.codex_strict_mode,
        load_embedding=use_rag,
    )
    RUNTIME = runtime

    if args.answer_strategy != "direct" and any(mode == "llm-only" for mode in args.modes):
        raise ValueError("mode=llm-only only supports --answer-strategy direct")

    rag: BiPharmRAG | None = None
    if use_rag:
        rag = BiPharmRAG(
            working_dir=str(args.working_dir),
            llm_model_func=local_llm_func,
            llm_model_name=args.llm_dir.name,
            llm_model_max_token_size=args.llm_max_token_size,
            embedding_func=EmbeddingFunc(
                embedding_dim=runtime.emb_dim,
                max_token_size=args.emb_max_length,
                func=local_embedding_func,
            ),
            enable_llm_cache=True,
        )
        maybe_insert_data(rag, args.data_file, args.working_dir)

    all_summaries: dict[str, dict[str, Any]] = {}
    config = {
        "benchmark_file": str(args.benchmark_file),
        "vocab_file": str(args.vocab_file),
        "case_file": str(args.case_file) if args.case_file else None,
        "working_dir": str(args.working_dir),
        "modes": args.modes,
        "num_questions": len(benchmark),
        "collect_context": args.collect_context,
        "answer_strategy": args.answer_strategy,
        "response_type": args.response_type,
        "top_k": args.top_k,
        "llm_backend": args.llm_backend,
        "llm_dir": str(args.llm_dir),
        "codex_model": args.codex_model if args.llm_backend == "codex-exec" else None,
        "codex_reasoning_effort": args.codex_reasoning_effort if args.llm_backend == "codex-exec" else None,
        "codex_workdir": str(args.codex_workdir) if args.llm_backend == "codex-exec" else None,
        "codex_audit_dir": str(args.codex_audit_dir) if args.llm_backend == "codex-exec" else None,
        "codex_strict_mode": args.codex_strict_mode if args.llm_backend == "codex-exec" else None,
        "emb_dir": str(args.emb_dir),
        "emb_device": args.emb_device,
        "time_started": int(time.time()),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for mode in args.modes:
        mode_rows: list[dict[str, Any]] = []
        print(f"\n=== Running mode={mode} on {len(benchmark)} questions ===", flush=True)
        for index, question in enumerate(benchmark, start=1):
            started = time.time()
            context = None
            raw_structured_response = None
            router_debug = None
            if mode == "llm-only":
                answer = direct_answer_without_retrieval(
                    runtime,
                    question["question"],
                    response_type=args.response_type,
                    max_new_tokens=args.structured_max_tokens,
                )
            elif args.answer_strategy == "structured":
                if rag is None:
                    raise RuntimeError("RAG runtime is required for structured retrieval modes.")
                context = rag.query(
                    question["question"],
                    QueryParam(
                        mode=mode,
                        only_need_context=True,
                        response_type=args.response_type,
                        top_k=args.top_k,
                    ),
                )
                answer, raw_structured_response = structured_answer_from_context(
                    runtime,
                    question,
                    context,
                    max_new_tokens=args.structured_max_tokens,
                )
            elif args.answer_strategy == "case-router":
                if rag is None:
                    raise RuntimeError("RAG runtime is required for case-router mode.")
                if case_router is None:
                    raise ValueError("--answer-strategy case-router requires --case-file or benchmark-adjacent cases.jsonl")
                context = rag.query(
                    question["question"],
                    QueryParam(
                        mode=mode,
                        only_need_context=True,
                        response_type=args.response_type,
                        top_k=args.top_k,
                    ),
                )
                answer, router_debug = case_router.route_case(question, context=context)
            elif args.answer_strategy == "casebridge":
                if rag is None:
                    raise RuntimeError("RAG runtime is required for casebridge mode.")
                if case_router is None:
                    raise ValueError("--answer-strategy casebridge requires --case-file or benchmark-adjacent cases.jsonl")
                raw_context = rag.query(
                    question["question"],
                    QueryParam(
                        mode=mode,
                        only_need_context=True,
                        response_type=args.response_type,
                        top_k=args.top_k,
                    ),
                )
                selected_case, router_debug = case_router.select_case(question, context=raw_context)
                if is_dual_bridge_question(question):
                    context = build_casebridge_context(question, raw_context, selected_case, case_router)
                    answer, raw_structured_response = casebridge_answer_from_context(
                        runtime,
                        question,
                        context,
                        max_new_tokens=args.structured_max_tokens,
                    )
                    if router_debug is None:
                        router_debug = {}
                    router_debug["bridge_context_chars"] = len(context)
                else:
                    answer = case_router.answer_from_case(question, selected_case) if selected_case is not None else ""
                    if not answer:
                        context = build_casebridge_context(question, raw_context, selected_case, case_router)
                        answer, raw_structured_response = structured_answer_from_context(
                            runtime,
                            question,
                            context,
                            max_new_tokens=args.structured_max_tokens,
                        )
                    else:
                        context = raw_context if args.collect_context else None
            else:
                if rag is None:
                    raise RuntimeError("RAG runtime is required for retrieval-backed direct mode.")
                param = QueryParam(
                    mode=mode,
                    response_type=args.response_type,
                    top_k=args.top_k,
                )
                answer = rag.query(question["question"], param)
                if args.collect_context:
                    context = rag.query(
                        question["question"],
                        QueryParam(
                            mode=mode,
                            only_need_context=True,
                            response_type=args.response_type,
                            top_k=args.top_k,
                        ),
                    )
            metrics = score_question(question, answer, vocab, context=context)
            row = {
                "mode": mode,
                "question_id": question["question_id"],
                "case_id": question["case_id"],
                "question_family": question["question_family"],
                "question_type": question["question_type"],
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "gold_answer_aliases": question["gold_answer_aliases"],
                "gold_slots": question["gold_slots"],
                "answer": answer,
                "context": context,
                "answer_strategy": args.answer_strategy,
                "raw_structured_response": raw_structured_response,
                "router_debug": router_debug,
                "elapsed_sec": round(time.time() - started, 3),
                "metrics": metrics,
            }
            mode_rows.append(row)
            print(
                f"[{mode}] {index}/{len(benchmark)} {question['question_id']} "
                f"hit={metrics['target_hit']} slot_recall={metrics['slot_recall']:.3f} "
                f"char_f1={metrics['char_f1']:.3f} elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )
            if index % 10 == 0 or index == len(benchmark):
                partial_summary = build_summary(mode_rows)
                write_jsonl(args.output_dir / f"{mode}_answers.jsonl", mode_rows)
                (args.output_dir / f"{mode}_summary.json").write_text(
                    json.dumps(partial_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        summary = build_summary(mode_rows)
        all_summaries[mode] = summary
        write_jsonl(args.output_dir / f"{mode}_answers.jsonl", mode_rows)
        (args.output_dir / f"{mode}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_summary_markdown(args.output_dir / f"{mode}_summary.md", mode, summary)
        write_examples_markdown(args.output_dir / f"{mode}_qualitative.md", mode_rows)

    (args.output_dir / "overall_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_overall_csv(args.output_dir / "overall_summary.csv", all_summaries)

    print(json.dumps(all_summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
