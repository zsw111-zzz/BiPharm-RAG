#!/usr/bin/env python3
"""Standalone BiPharm-HG evaluation for TCM pathology and bridge benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import run_tcm_piwei_eval as base_eval
import build_tcm_piwei_pharmacology_bridge as bridge_utils


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MODEL_ROOT = first_existing_path(
    REPO_ROOT / "models",
    WORKSPACE_ROOT / "models",
)
DEFAULT_CODEX_ROOT = REPO_ROOT / ".codex_tmp"


DOSAGE_PATTERN = re.compile(r"(?:用法与用量|用量)[:：]\s*([^。；]+)")
CAUTION_PATTERN = re.compile(r"(?:禁忌|注意|慎用|不宜)[:：]\s*([^。；]+)")
HERB_NAME_PATTERN = re.compile(r"药材[:：]\s*([^。；]+)")
EFFICACY_PATTERN = re.compile(r"功效[:：]\s*([^。；]+)")
INDICATIONS_PATTERN = re.compile(r"主治[:：]\s*([^。；]+)")
CLUE_FIELD_LABELS = {
    "efficacy": "功效",
    "indications": "主治",
    "dosage": "用法与用量",
    "cautions": "禁忌/注意",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return base_eval.load_jsonl(path)


def load_benchmark_stats(benchmark_dir: Path) -> dict[str, Any]:
    stats_file = benchmark_dir / "stats.json"
    if not stats_file.exists():
        return {}
    try:
        return json.loads(stats_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_local_artifact_from_stats(
    benchmark_dir: Path,
    stats_key: str,
    expected_name: str,
) -> Path | None:
    stats = load_benchmark_stats(benchmark_dir)
    raw_path = stats.get(stats_key)
    if not raw_path:
        return None
    raw_path = Path(str(raw_path))
    sibling_dir = benchmark_dir.parent / raw_path.parent.name
    candidates = [
        sibling_dir / expected_name,
        sibling_dir / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_case_file_for_benchmark(benchmark_file: Path) -> Path | None:
    benchmark_dir = benchmark_file.parent
    direct_case_file = benchmark_dir / "cases.jsonl"
    if direct_case_file.exists():
        return direct_case_file
    stats_case_file = resolve_local_artifact_from_stats(benchmark_dir, "case_file", "cases.jsonl")
    if stats_case_file is not None:
        return stats_case_file
    benchmark_name = benchmark_dir.name
    if "_dual_reverse_" in benchmark_name:
        sibling_candidates = [benchmark_name.replace("_dual_reverse_", "_", 1)]
    elif "_dual_" in benchmark_name:
        sibling_candidates = [benchmark_name.replace("_dual_", "_", 1)]
    else:
        sibling_candidates = []
    if sibling_candidates:
        sibling_candidates.extend(
            [
                re.sub(r"_herb_implicit.*$", "", sibling_candidates[0]),
                re.sub(r"_formula_alias_perturb.*$", "", sibling_candidates[0]),
            ]
        )
        seen: set[str] = set()
        for sibling_name in sibling_candidates:
            sibling_name = sibling_name.strip()
            if not sibling_name or sibling_name in seen:
                continue
            seen.add(sibling_name)
            sibling_case_file = benchmark_dir.parent / sibling_name / "cases.jsonl"
            if sibling_case_file.exists():
                return sibling_case_file
    return None


def default_bridge_file_for_benchmark(benchmark_file: Path) -> Path | None:
    benchmark_dir = benchmark_file.parent
    direct_bridge_file = benchmark_dir / "bridge_herb_corpus.jsonl"
    if direct_bridge_file.exists():
        return direct_bridge_file
    stats_bridge_file = resolve_local_artifact_from_stats(
        benchmark_dir,
        "bridge_corpus_file",
        "bridge_herb_corpus.jsonl",
    )
    if stats_bridge_file is not None:
        return stats_bridge_file
    benchmark_name = benchmark_dir.name
    if "_dual_reverse_" in benchmark_name:
        sibling_candidates = [benchmark_name.replace("_dual_reverse_", "_dual_", 1)]
    else:
        sibling_candidates = []
    for sibling_name in sibling_candidates:
        sibling_bridge_file = benchmark_dir.parent / sibling_name / "bridge_herb_corpus.jsonl"
        if sibling_bridge_file.exists():
            return sibling_bridge_file
    return None


def render_pathology_memory(
    question: dict[str, Any],
    selected_case: dict[str, Any] | None,
    case_router: base_eval.CaseRouter | None,
) -> str:
    lines: list[str] = ["【病理图锚点】"]
    syndrome = base_eval.question_syndrome(question)
    formula_name = base_eval.question_formula(question)
    condition = base_eval.question_condition(question)
    herb_terms = base_eval.question_herb_terms(
        question,
        include_structured_fields=not is_reverse_bridge_question(question),
    )
    symptoms = base_eval.parse_question_symptoms(question["question"])

    if symptoms:
        lines.append(f"症状: {symptoms}")
    if syndrome:
        lines.append(f"证型: {syndrome}")
    if formula_name:
        lines.append(f"方剂: {formula_name}")
    if condition:
        lines.append(f"加减条件: {condition}")
    if herb_terms:
        lines.append(f"目标药味: {' / '.join(herb_terms)}")

    if selected_case is None:
        lines.append("未命中病例节点。")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "【命中病理节点】",
            f"case_id: {selected_case.get('case_id', '')}",
            f"证型: {str(selected_case.get('syndrome', '')).strip()}",
            f"病机: {str(selected_case.get('pathogenesis', '')).strip()}",
            f"治法: {str(selected_case.get('therapy', '')).strip()}",
            f"方剂: {str(selected_case.get('formula_name', '')).strip()}",
            f"方药组成: {str(selected_case.get('formula_text', '')).strip()}",
        ]
    )
    if condition and case_router is not None:
        mod, exact_match, mod_score = case_router.best_modification(selected_case, condition)
        if mod is not None:
            herbs = [str(item).strip() for item in mod.get("herbs", []) if str(item).strip()]
            lines.extend(
                [
                    "",
                    "【加减条件节点】",
                    f"匹配条件: {str(mod.get('condition', '')).strip()}",
                    f"精确匹配: {'yes' if exact_match else 'no'}",
                    f"条件得分: {mod_score:.4f}",
                    f"加减药味: {'、'.join(herbs)}",
                    f"原文: {str(mod.get('raw_clause', '')).strip()}",
                ]
            )
    return "\n".join(lines)


def render_pathology_unstructured_case(
    selected_case: dict[str, Any] | None,
) -> str:
    lines = ["【病例原文片段】"]
    if selected_case is None:
        lines.append("未命中病例原文。")
        return "\n".join(lines)

    excerpt = str(selected_case.get("raw_text_excerpt", "")).strip()
    formula_text = str(selected_case.get("formula_text", "")).strip()
    if excerpt:
        lines.append(excerpt)
    if formula_text:
        lines.append(formula_text)
    for mod in selected_case.get("modification_pairs", []):
        raw_clause = str(mod.get("raw_clause", "")).strip()
        if raw_clause:
            lines.append(raw_clause)
    return "\n".join(lines)


def select_case_for_question(
    question: dict[str, Any],
    case_router: base_eval.CaseRouter,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    return case_router.select_case(question, context=question["question"])


def question_clue_field(question: dict[str, Any]) -> str:
    return str(question.get("clue_field", "")).strip()


def question_pharmacology_clue(question: dict[str, Any]) -> str:
    return str(question.get("pharmacology_clue", "")).strip()


def is_reverse_bridge_question(question: dict[str, Any]) -> bool:
    return question.get("question_type") in {"herb_from_formula_clue", "herb_from_condition_clue"}


@dataclass
class HerbEntry:
    herb_name: str
    efficacy: str
    indications: str
    dosage: str
    cautions: str
    raw_text: str
    payload: dict[str, Any]


class HerbGraph:
    """Pharmacopoeia-side herb graph used by BiPharm-HG bridge evaluation."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        formula_weight: float = 25.0,
        condition_weight: float = 40.0,
        syndrome_weight: float = 10.0,
        use_herb_name_prior: bool = True,
        exact_herb_match_weight: float = 100.0,
        partial_herb_match_weight: float = 40.0,
    ) -> None:
        self.rows = rows
        self.formula_weight = formula_weight
        self.condition_weight = condition_weight
        self.syndrome_weight = syndrome_weight
        self.use_herb_name_prior = use_herb_name_prior
        self.exact_herb_match_weight = exact_herb_match_weight
        self.partial_herb_match_weight = partial_herb_match_weight
        self.entries: list[HerbEntry] = [self._build_entry(row) for row in rows]
        self.by_name: dict[str, list[HerbEntry]] = {}
        for entry in self.entries:
            key = base_eval.normalize_answer(entry.herb_name)
            if not key:
                continue
            self.by_name.setdefault(key, []).append(entry)

    @staticmethod
    def _extract(pattern: re.Pattern[str], text: str) -> str:
        match = pattern.search(text)
        return match.group(1).strip() if match else ""

    def _build_entry(self, row: dict[str, Any]) -> HerbEntry:
        text = str(row.get("text", "")).strip()
        herb_name = self._extract(HERB_NAME_PATTERN, text)
        herb_name = herb_name or str(row.get("metadata", {}).get("matched_pharm_herb", "")).strip()
        return HerbEntry(
            herb_name=herb_name,
            efficacy=self._extract(EFFICACY_PATTERN, text),
            indications=self._extract(INDICATIONS_PATTERN, text),
            dosage=self._extract(DOSAGE_PATTERN, text),
            cautions=self._extract(CAUTION_PATTERN, text),
            raw_text=text,
            payload=row,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        formula_weight: float = 25.0,
        condition_weight: float = 40.0,
        syndrome_weight: float = 10.0,
        use_herb_name_prior: bool = True,
        exact_herb_match_weight: float = 100.0,
        partial_herb_match_weight: float = 40.0,
    ) -> "HerbGraph":
        return cls(
            load_jsonl(path),
            formula_weight=formula_weight,
            condition_weight=condition_weight,
            syndrome_weight=syndrome_weight,
            use_herb_name_prior=use_herb_name_prior,
            exact_herb_match_weight=exact_herb_match_weight,
            partial_herb_match_weight=partial_herb_match_weight,
        )

    def _bridge_score(
        self,
        entry: HerbEntry,
        question: dict[str, Any],
        selected_case: dict[str, Any] | None,
        case_router: base_eval.CaseRouter | None,
    ) -> tuple[float, dict[str, Any]]:
        herb_terms = base_eval.question_herb_terms(question)
        herb_norm = base_eval.normalize_answer(entry.herb_name)
        score = 0.0
        debug: dict[str, Any] = {
            "selected_herb": entry.herb_name,
            "formula_bridge_hit": False,
            "condition_bridge_hit": False,
            "syndrome_bridge_hit": False,
            "dosage_present": bool(entry.dosage),
            "cautions_present": bool(entry.cautions),
        }

        for term in herb_terms:
            term_norm = base_eval.normalize_answer(term)
            if not term_norm:
                continue
            if herb_norm == term_norm:
                score += self.exact_herb_match_weight
            elif herb_norm in term_norm or term_norm in herb_norm:
                score += self.partial_herb_match_weight

        if selected_case is not None:
            syndrome = str(selected_case.get("syndrome", "")).strip()
            question_syndrome = base_eval.question_syndrome(question)
            if syndrome and question_syndrome and base_eval.normalize_answer(syndrome) == base_eval.normalize_answer(question_syndrome):
                score += self.syndrome_weight
                debug["syndrome_bridge_hit"] = True

            if case_router is not None and case_router.case_contains_herb_terms(selected_case, herb_terms):
                score += self.formula_weight
                debug["formula_bridge_hit"] = True

            if (
                question["question_type"] == "herb_indications_from_condition"
                and case_router is not None
                and base_eval.question_condition(question)
            ):
                mod, exact_match, mod_score = case_router.best_modification(selected_case, base_eval.question_condition(question))
                if mod is not None:
                    score += mod_score * 15.0
                    if case_router.modification_contains_herb_terms(mod, herb_terms):
                        score += self.condition_weight
                        debug["condition_bridge_hit"] = True
                    if exact_match:
                        score += 20.0

        if question["question_type"] == "herb_efficacy_from_case" and entry.efficacy:
            score += 8.0
        if question["question_type"] == "herb_indications_from_condition" and entry.indications:
            score += 8.0

        return score, debug

    def select_entry(
        self,
        question: dict[str, Any],
        selected_case: dict[str, Any] | None,
        case_router: base_eval.CaseRouter | None,
    ) -> tuple[HerbEntry | None, dict[str, Any]]:
        herb_terms = base_eval.question_herb_terms(question)
        candidates: list[HerbEntry] = []
        seen: set[str] = set()
        for term in herb_terms:
            term_norm = base_eval.normalize_answer(term)
            if not term_norm:
                continue
            for entry in self.by_name.get(term_norm, []):
                key = f"{entry.herb_name}::{entry.payload.get('record_id', '')}"
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(entry)
        if not self.use_herb_name_prior or not candidates:
            candidates = self.entries

        if not candidates:
            return None, {"selected_herb": None, "num_candidates": 0, "candidate_scores": []}

        scored: list[tuple[float, HerbEntry, dict[str, Any]]] = []
        for entry in candidates:
            score, debug = self._bridge_score(entry, question, selected_case, case_router)
            scored.append((score, entry, debug))
        scored.sort(key=lambda item: (item[0], item[1].herb_name), reverse=True)
        best_score, best_entry, best_debug = scored[0]
        best_debug = dict(best_debug)
        best_debug["bridge_score"] = round(best_score, 4)
        best_debug["num_candidates"] = len(candidates)
        best_debug["candidate_scores"] = [
            {
                "herb": item[1].herb_name,
                "score": round(item[0], 4),
                **item[2],
            }
            for item in scored[:3]
        ]
        return best_entry, best_debug

    def lookup_entries_by_terms(self, herb_terms: list[str]) -> list[HerbEntry]:
        candidates: list[HerbEntry] = []
        seen: set[str] = set()
        for term in herb_terms:
            term_norm = base_eval.normalize_answer(term)
            if not term_norm:
                continue
            for entry in self.by_name.get(term_norm, []):
                key = f"{entry.herb_name}::{entry.payload.get('record_id', '')}"
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(entry)

        if not candidates:
            for entry in self.entries:
                herb_norm = base_eval.normalize_answer(entry.herb_name)
                for term in herb_terms:
                    term_norm = base_eval.normalize_answer(term)
                    if not term_norm:
                        continue
                    if term_norm in herb_norm or herb_norm in term_norm:
                        key = f"{entry.herb_name}::{entry.payload.get('record_id', '')}"
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(entry)
                        break

        candidates.sort(key=lambda entry: (len(base_eval.normalize_answer(entry.herb_name)), entry.herb_name))
        return candidates

    def lookup_entry_by_herb_terms(
        self,
        question: dict[str, Any],
    ) -> tuple[HerbEntry | None, dict[str, Any]]:
        herb_terms = base_eval.question_herb_terms(question)
        candidates = self.lookup_entries_by_terms(herb_terms)

        if not candidates:
            return None, {"selected_herb": None, "num_candidates": 0, "candidate_scores": [], "lookup_mode": True}
        best_entry = candidates[0]
        debug = {
            "lookup_mode": True,
            "selected_herb": best_entry.herb_name,
            "num_candidates": len(candidates),
            "candidate_scores": [
                {
                    "herb": entry.herb_name,
                    "score": 1.0 if index == 0 else 0.0,
                }
                for index, entry in enumerate(candidates[:3])
            ],
        }
        return best_entry, debug

    @staticmethod
    def field_text(entry: HerbEntry, clue_field: str) -> str:
        if clue_field == "efficacy":
            return entry.efficacy
        if clue_field == "indications":
            return entry.indications
        if clue_field == "dosage":
            return entry.dosage
        if clue_field == "cautions":
            return entry.cautions
        return ""

    def select_entry_by_clue(
        self,
        clue_field: str,
        clue_text: str,
        *,
        candidates: list[HerbEntry] | None = None,
    ) -> tuple[HerbEntry | None, dict[str, Any]]:
        pool = candidates if candidates is not None else self.entries
        if not pool:
            return None, {"selected_herb": None, "num_candidates": 0, "candidate_scores": []}

        clue_norm = base_eval.normalize_answer(clue_text)
        scored: list[tuple[float, HerbEntry, dict[str, Any]]] = []
        for entry in pool:
            field_text = self.field_text(entry, clue_field)
            field_norm = base_eval.normalize_answer(field_text)
            exact_hit = bool(clue_norm and field_norm and clue_norm in field_norm)
            score = 0.0
            if exact_hit:
                score += 100.0 + min(len(clue_norm), 12)
            score += base_eval.char_f1(clue_text, field_text) * 20.0
            debug = {
                "selected_herb": entry.herb_name,
                "clue_field": clue_field,
                "clue_text": clue_text,
                "field_text": field_text,
                "clue_exact_hit": exact_hit,
            }
            scored.append((score, entry, debug))

        scored.sort(key=lambda item: (item[0], item[1].herb_name), reverse=True)
        best_score, best_entry, best_debug = scored[0]
        best_debug = dict(best_debug)
        best_debug["bridge_score"] = round(best_score, 4)
        best_debug["num_candidates"] = len(pool)
        best_debug["candidate_scores"] = [
            {
                "herb": item[1].herb_name,
                "score": round(item[0], 4),
                "clue_exact_hit": item[2]["clue_exact_hit"],
                "field_text": item[2]["field_text"],
            }
            for item in scored[:5]
        ]
        return best_entry, best_debug

    def select_entry_by_raw_text_clue(
        self,
        clue_text: str,
        *,
        candidates: list[HerbEntry] | None = None,
    ) -> tuple[HerbEntry | None, dict[str, Any]]:
        pool = candidates if candidates is not None else self.entries
        if not pool:
            return None, {"selected_herb": None, "num_candidates": 0, "candidate_scores": []}

        clue_norm = base_eval.normalize_answer(clue_text)
        scored: list[tuple[float, HerbEntry, dict[str, Any]]] = []
        for entry in pool:
            match_text = entry.raw_text
            match_norm = base_eval.normalize_answer(match_text)
            exact_hit = bool(clue_norm and match_norm and clue_norm in match_norm)
            score = 0.0
            if exact_hit:
                score += 100.0 + min(len(clue_norm), 12)
            score += base_eval.char_f1(clue_text, match_text) * 20.0
            debug = {
                "selected_herb": entry.herb_name,
                "clue_text": clue_text,
                "match_text": match_text,
                "clue_exact_hit": exact_hit,
                "control_mode": "local_nongraph_raw_text",
            }
            scored.append((score, entry, debug))

        scored.sort(key=lambda item: (item[0], item[1].herb_name), reverse=True)
        best_score, best_entry, best_debug = scored[0]
        best_debug = dict(best_debug)
        best_debug["bridge_score"] = round(best_score, 4)
        best_debug["num_candidates"] = len(pool)
        best_debug["candidate_scores"] = [
            {
                "herb": item[1].herb_name,
                "score": round(item[0], 4),
                "clue_exact_hit": item[2]["clue_exact_hit"],
                "match_text": item[2]["match_text"],
            }
            for item in scored[:5]
        ]
        return best_entry, best_debug


def build_bipharm_context(
    question: dict[str, Any],
    selected_case: dict[str, Any] | None,
    case_router: base_eval.CaseRouter | None,
    herb_entry: HerbEntry | None,
    bridge_debug: dict[str, Any] | None,
) -> str:
    lines = [render_pathology_memory(question, selected_case, case_router), "", "【桥接边审计】"]
    bridge_type = str(question.get("bridge_type", "")).strip()
    if bridge_type:
        lines.append(f"桥接类型: {bridge_type}")
    if bridge_debug:
        lines.append(f"方剂-药味命中: {bridge_debug.get('formula_bridge_hit')}")
        lines.append(f"加减条件-药味命中: {bridge_debug.get('condition_bridge_hit')}")
        lines.append(f"证型锚定命中: {bridge_debug.get('syndrome_bridge_hit')}")
        lines.append(f"桥接得分: {bridge_debug.get('bridge_score')}")

    lines.extend(["", "【药理图节点】"])
    if herb_entry is None:
        lines.append("未命中药理节点。")
        return "\n".join(lines)

    lines.append(f"药材: {herb_entry.herb_name}")
    if herb_entry.efficacy:
        lines.append(f"功效: {herb_entry.efficacy}")
    if herb_entry.indications:
        lines.append(f"主治: {herb_entry.indications}")
    if herb_entry.dosage:
        lines.append(f"用法与用量: {herb_entry.dosage}")
    if herb_entry.cautions:
        lines.append(f"禁忌/注意: {herb_entry.cautions}")
    lines.append(f"原始证据: {herb_entry.raw_text}")
    return "\n".join(lines)


def build_lookup_context(question: dict[str, Any], herb_entry: HerbEntry | None) -> str:
    herb_terms = base_eval.question_herb_terms(question)
    lines = ["【药典直查控制】"]
    if herb_terms:
        lines.append(f"题面给定药味: {' / '.join(herb_terms)}")
    if herb_entry is None:
        lines.append("未命中药典条目。")
        return "\n".join(lines)

    lines.append(f"命中药材: {herb_entry.herb_name}")
    if herb_entry.efficacy:
        lines.append(f"功效: {herb_entry.efficacy}")
    if herb_entry.indications:
        lines.append(f"主治: {herb_entry.indications}")
    if herb_entry.dosage:
        lines.append(f"用法与用量: {herb_entry.dosage}")
    if herb_entry.cautions:
        lines.append(f"禁忌/注意: {herb_entry.cautions}")
    lines.append(f"原始证据: {herb_entry.raw_text}")
    return "\n".join(lines)


def resolve_reverse_candidate_entries(
    question: dict[str, Any],
    selected_case: dict[str, Any] | None,
    case_router: base_eval.CaseRouter | None,
    herb_graph: HerbGraph,
    *,
    candidate_strategy: str = "default",
) -> tuple[list[HerbEntry], dict[str, Any]]:
    debug: dict[str, Any] = {
        "candidate_source": None,
        "candidate_strategy": candidate_strategy,
        "candidate_pathology_herbs": [],
        "candidate_pharm_herbs": [],
        "condition_exact_match": None,
        "condition_score": None,
    }
    if selected_case is None:
        return [], debug

    formula_terms = bridge_utils.extract_formula_herbs(selected_case.get("formula_text", ""))
    modification_terms: list[str] = []
    if question["question_type"] == "herb_from_condition_clue":
        condition = base_eval.question_condition(question)
        if case_router is not None and condition:
            mod, exact_match, mod_score = case_router.best_modification(selected_case, condition)
            if mod is not None:
                modification_terms = [str(item).strip() for item in mod.get("herbs", []) if str(item).strip()]
            debug["condition_exact_match"] = exact_match
            debug["condition_score"] = round(mod_score, 4)

    raw_terms: list[str] = []
    if candidate_strategy == "default":
        if question["question_type"] == "herb_from_formula_clue":
            raw_terms = formula_terms
            debug["candidate_source"] = "formula_text"
        elif question["question_type"] == "herb_from_condition_clue":
            raw_terms = modification_terms
            debug["candidate_source"] = "modification_pairs"
    elif candidate_strategy == "formula_only":
        raw_terms = formula_terms
        debug["candidate_source"] = "formula_text"
    elif candidate_strategy == "formula_plus_modification":
        raw_terms = formula_terms + modification_terms
        debug["candidate_source"] = "formula_plus_modification"
    else:
        raise ValueError(f"Unsupported reverse candidate strategy: {candidate_strategy}")

    candidates: list[HerbEntry] = []
    seen: set[str] = set()
    for raw_term in raw_terms:
        matched_entries = herb_graph.lookup_entries_by_terms([raw_term])
        if not matched_entries:
            continue
        best_entry = matched_entries[0]
        herb_norm = base_eval.normalize_answer(best_entry.herb_name)
        if herb_norm in seen:
            continue
        seen.add(herb_norm)
        candidates.append(best_entry)
        debug["candidate_pathology_herbs"].append(raw_term)
        debug["candidate_pharm_herbs"].append(best_entry.herb_name)
    return candidates, debug


def build_reverse_bridge_context(
    question: dict[str, Any],
    selected_case: dict[str, Any] | None,
    case_router: base_eval.CaseRouter | None,
    candidate_entries: list[HerbEntry],
    herb_entry: HerbEntry | None,
    bridge_debug: dict[str, Any] | None,
    candidate_debug: dict[str, Any] | None,
) -> str:
    lines = [render_pathology_memory(question, selected_case, case_router), "", "【反向桥接任务】"]
    clue_field = question_clue_field(question)
    clue_text = question_pharmacology_clue(question)
    if clue_field:
        lines.append(f"药典线索字段: {CLUE_FIELD_LABELS.get(clue_field, clue_field)}")
    if clue_text:
        lines.append(f"药典线索: {clue_text}")
    if candidate_debug:
        raw_terms = candidate_debug.get("candidate_pathology_herbs") or []
        pharm_terms = candidate_debug.get("candidate_pharm_herbs") or []
        if raw_terms:
            lines.append(f"病理侧候选药味: {'、'.join(raw_terms)}")
        if pharm_terms:
            lines.append(f"药典侧候选药材: {'、'.join(pharm_terms)}")

    lines.extend(["", "【桥接边审计】"])
    if bridge_debug:
        lines.append(f"桥接得分: {bridge_debug.get('bridge_score')}")
        lines.append(f"候选数: {bridge_debug.get('num_candidates')}")
    if candidate_debug and candidate_debug.get("condition_exact_match") is not None:
        lines.append(f"加减条件精确命中: {candidate_debug.get('condition_exact_match')}")
        lines.append(f"加减条件得分: {candidate_debug.get('condition_score')}")

    lines.extend(["", "【药理图节点】"])
    if herb_entry is None:
        lines.append("未命中药理节点。")
        return "\n".join(lines)

    lines.append(f"药材: {herb_entry.herb_name}")
    if herb_entry.efficacy:
        lines.append(f"功效: {herb_entry.efficacy}")
    if herb_entry.indications:
        lines.append(f"主治: {herb_entry.indications}")
    if herb_entry.dosage:
        lines.append(f"用法与用量: {herb_entry.dosage}")
    if herb_entry.cautions:
        lines.append(f"禁忌/注意: {herb_entry.cautions}")
    lines.append(f"原始证据: {herb_entry.raw_text}")
    return "\n".join(lines)


def build_clue_only_context(
    question: dict[str, Any],
    herb_entry: HerbEntry | None,
    bridge_debug: dict[str, Any] | None,
) -> str:
    lines = ["【药典线索控制】"]
    clue_field = question_clue_field(question)
    clue_text = question_pharmacology_clue(question)
    if clue_field:
        lines.append(f"线索字段: {CLUE_FIELD_LABELS.get(clue_field, clue_field)}")
    if clue_text:
        lines.append(f"线索文本: {clue_text}")
    if bridge_debug:
        lines.append(f"候选数: {bridge_debug.get('num_candidates')}")
    if herb_entry is None:
        lines.append("未命中药典条目。")
        return "\n".join(lines)

    lines.append(f"命中药材: {herb_entry.herb_name}")
    if herb_entry.efficacy:
        lines.append(f"功效: {herb_entry.efficacy}")
    if herb_entry.indications:
        lines.append(f"主治: {herb_entry.indications}")
    lines.append(f"原始证据: {herb_entry.raw_text}")
    return "\n".join(lines)


def bridge_answer_from_entry(question: dict[str, Any], herb_entry: HerbEntry | None) -> str:
    if herb_entry is None:
        return ""
    if is_reverse_bridge_question(question):
        return herb_entry.herb_name
    if question["question_type"] == "herb_efficacy_from_case":
        return herb_entry.efficacy
    if question["question_type"] == "herb_indications_from_condition":
        return herb_entry.indications
    return ""


def run_bipharm_diag(
    runtime: base_eval.LocalRuntime | None,
    question: dict[str, Any],
    case_router: base_eval.CaseRouter,
    max_new_tokens: int,
    *,
    force_generation: bool = False,
) -> tuple[str, str, dict[str, Any], str | None]:
    selected_case, router_debug = select_case_for_question(question, case_router)
    context = render_pathology_memory(question, selected_case, case_router)
    answer = ""
    if not force_generation and selected_case is not None:
        answer = case_router.answer_from_case(question, selected_case)
    raw_structured_response = None
    if not answer and runtime is not None:
        if force_generation:
            context = render_pathology_unstructured_case(selected_case)
        answer, raw_structured_response = base_eval.structured_answer_from_context(
            runtime,
            question,
            context,
            max_new_tokens=max_new_tokens,
        )
    if router_debug is None:
        router_debug = {}
    if force_generation:
        router_debug["ablation_note"] = "slot_aware_realization_disabled"
    return answer, context, router_debug, raw_structured_response


def run_bipharm_hg(
    runtime: base_eval.LocalRuntime | None,
    question: dict[str, Any],
    case_router: base_eval.CaseRouter,
    herb_graph: HerbGraph,
    max_new_tokens: int,
    *,
    force_bridge_generation: bool = False,
) -> tuple[str, str, dict[str, Any], str | None]:
    selected_case, router_debug = select_case_for_question(question, case_router)
    herb_entry, bridge_debug = herb_graph.select_entry(question, selected_case, case_router)
    context = build_bipharm_context(question, selected_case, case_router, herb_entry, bridge_debug)
    router_debug = dict(router_debug)
    router_debug["bridge_debug"] = bridge_debug
    answer = "" if force_bridge_generation else bridge_answer_from_entry(question, herb_entry)
    raw_structured_response = None
    if not answer and runtime is not None:
        answer, raw_structured_response = base_eval.casebridge_answer_from_context(
            runtime,
            question,
            context,
            max_new_tokens=max_new_tokens,
        )
    return answer, context, router_debug, raw_structured_response


def run_pharm_lookup(
    question: dict[str, Any],
    herb_graph: HerbGraph,
) -> tuple[str, str, dict[str, Any], None]:
    herb_entry, lookup_debug = herb_graph.lookup_entry_by_herb_terms(question)
    context = build_lookup_context(question, herb_entry)
    answer = bridge_answer_from_entry(question, herb_entry)
    return answer, context, {"bridge_debug": lookup_debug}, None


def run_bipharm_hg_reverse(
    question: dict[str, Any],
    case_router: base_eval.CaseRouter,
    herb_graph: HerbGraph,
    *,
    candidate_strategy: str = "default",
) -> tuple[str, str, dict[str, Any], None]:
    selected_case, router_debug = select_case_for_question(question, case_router)
    candidate_entries, candidate_debug = resolve_reverse_candidate_entries(
        question,
        selected_case,
        case_router,
        herb_graph,
        candidate_strategy=candidate_strategy,
    )
    herb_entry, bridge_debug = herb_graph.select_entry_by_clue(
        question_clue_field(question),
        question_pharmacology_clue(question),
        candidates=candidate_entries,
    )
    context = build_reverse_bridge_context(
        question,
        selected_case,
        case_router,
        candidate_entries,
        herb_entry,
        bridge_debug,
        candidate_debug,
    )
    router_debug = dict(router_debug)
    router_debug["bridge_debug"] = bridge_debug
    router_debug["reverse_candidate_debug"] = candidate_debug
    answer = bridge_answer_from_entry(question, herb_entry)
    return answer, context, router_debug, None


def run_pharm_clue_only(
    question: dict[str, Any],
    herb_graph: HerbGraph,
) -> tuple[str, str, dict[str, Any], None]:
    herb_entry, bridge_debug = herb_graph.select_entry_by_clue(
        question_clue_field(question),
        question_pharmacology_clue(question),
    )
    context = build_clue_only_context(question, herb_entry, bridge_debug)
    answer = bridge_answer_from_entry(question, herb_entry)
    return answer, context, {"bridge_debug": bridge_debug}, None


def build_local_nongraph_reverse_context(
    question: dict[str, Any],
    selected_case: dict[str, Any] | None,
    case_router: base_eval.CaseRouter | None,
    herb_entry: HerbEntry | None,
    bridge_debug: dict[str, Any] | None,
    candidate_debug: dict[str, Any] | None,
) -> str:
    lines = [render_pathology_memory(question, selected_case, case_router), "", "【同局部候选集非图控制】"]
    clue_field = question_clue_field(question)
    clue_text = question_pharmacology_clue(question)
    if clue_field:
        lines.append(f"线索字段: {CLUE_FIELD_LABELS.get(clue_field, clue_field)}")
    if clue_text:
        lines.append(f"线索文本: {clue_text}")
    if candidate_debug:
        raw_terms = candidate_debug.get("candidate_pathology_herbs") or []
        pharm_terms = candidate_debug.get("candidate_pharm_herbs") or []
        if raw_terms:
            lines.append(f"病理侧候选药味: {'、'.join(raw_terms)}")
        if pharm_terms:
            lines.append(f"局部候选药材: {'、'.join(pharm_terms)}")
    if bridge_debug:
        lines.append(f"候选数: {bridge_debug.get('num_candidates')}")
        lines.append("控制策略: 忽略字段型图结构，仅对局部候选药材全文做线索匹配")
    if herb_entry is None:
        lines.append("未命中药理节点。")
        return "\n".join(lines)

    lines.extend(["", "【命中药理节点】"])
    lines.append(f"药材: {herb_entry.herb_name}")
    lines.append(f"原始证据: {herb_entry.raw_text}")
    return "\n".join(lines)


def run_local_nongraph_reverse(
    question: dict[str, Any],
    case_router: base_eval.CaseRouter,
    herb_graph: HerbGraph,
    *,
    candidate_strategy: str = "default",
) -> tuple[str, str, dict[str, Any], None]:
    selected_case, router_debug = select_case_for_question(question, case_router)
    candidate_entries, candidate_debug = resolve_reverse_candidate_entries(
        question,
        selected_case,
        case_router,
        herb_graph,
        candidate_strategy=candidate_strategy,
    )
    herb_entry, bridge_debug = herb_graph.select_entry_by_raw_text_clue(
        question_pharmacology_clue(question),
        candidates=candidate_entries,
    )
    context = build_local_nongraph_reverse_context(
        question,
        selected_case,
        case_router,
        herb_entry,
        bridge_debug,
        candidate_debug,
    )
    router_debug = dict(router_debug)
    router_debug["bridge_debug"] = bridge_debug
    router_debug["reverse_candidate_debug"] = candidate_debug
    router_debug["ablation_note"] = "same_local_candidate_nongraph_control"
    answer = bridge_answer_from_entry(question, herb_entry)
    return answer, context, router_debug, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--vocab-file", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, default=None)
    parser.add_argument("--bridge-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["llm-only", "bipharm-diag"])
    parser.add_argument("--question-types", nargs="*", default=None)
    parser.add_argument("--question-families", nargs="*", default=None)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--response-type", type=str, default="请用紧凑中文作答，优先给出原文短语或按题目要求列点，不要展开。")
    parser.add_argument("--structured-max-tokens", type=int, default=256)
    parser.add_argument("--llm-dir", type=Path, default=DEFAULT_MODEL_ROOT / "Qwen" / "Qwen3.5-4B")
    parser.add_argument("--llm-backend", choices=["local", "codex-exec"], default="local")
    parser.add_argument("--codex-model", type=str, default="gpt-5.4")
    parser.add_argument("--codex-reasoning-effort", choices=["low", "medium", "high", "xhigh"], default="low")
    parser.add_argument("--codex-workdir", type=Path, default=DEFAULT_CODEX_ROOT)
    parser.add_argument("--codex-audit-dir", type=Path, default=DEFAULT_CODEX_ROOT / "audit")
    parser.add_argument(
        "--codex-strict-mode",
        action="store_true",
        help="Enable benchmark-only hard checks for codex-exec JSON stream, stderr, and tool usage.",
    )
    parser.add_argument("--emb-dir", type=Path, default=DEFAULT_MODEL_ROOT / "BAAI" / "bge-m3")
    parser.add_argument("--llm-device-map", type=str, default="auto")
    parser.add_argument("--llm-dtype", type=str, default="float16")
    parser.add_argument("--llm-max-token-size", type=int, default=8192)
    parser.add_argument("--emb-device", type=str, default="cuda:0")
    parser.add_argument("--emb-max-length", type=int, default=2048)
    parser.add_argument("--emb-batch-size", type=int, default=16)
    parser.add_argument("--disable-llm-fallback", action="store_true")
    parser.add_argument("--formula-bridge-weight", type=float, default=25.0)
    parser.add_argument("--condition-bridge-weight", type=float, default=40.0)
    parser.add_argument("--syndrome-bridge-weight", type=float, default=10.0)
    parser.add_argument("--disable-herb-name-prior", action="store_true")
    parser.add_argument("--exact-herb-match-weight", type=float, default=100.0)
    parser.add_argument("--partial-herb-match-weight", type=float, default=40.0)
    parser.add_argument("--force-bridge-generation", action="store_true")
    args = parser.parse_args()

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
        args.case_file = default_case_file_for_benchmark(args.benchmark_file)

    case_router = None
    pathology_modes = {
        "bipharm-diag",
        "bipharm-diag-wo-typed-memory",
        "bipharm-diag-wo-slot-realization",
        "bipharm-diag-wo-modification-routing",
    }
    bridge_case_modes = {
        "bipharm-hg",
        "bipharm-hg-reverse",
        "local-nongraph-reverse",
        "local-nongraph-formula-only-reverse",
        "local-nongraph-formula-plus-mod-reverse",
    }
    if any(mode in pathology_modes | bridge_case_modes for mode in args.modes):
        if args.case_file is None or not args.case_file.exists():
            raise ValueError("BiPharm-HG modes require a case file.")
        case_routers = {
            "full": base_eval.CaseRouter.load(args.case_file),
            "wo_typed_memory": base_eval.CaseRouter.load(args.case_file, use_typed_case_memory=False),
            "wo_modification_routing": base_eval.CaseRouter.load(args.case_file, enable_modification_routing=False),
        }
    else:
        case_routers = {}

    herb_graph = None
    if any(
        mode in {
            "bipharm-hg",
            "pharm-lookup",
            "bipharm-hg-reverse",
            "pharm-clue-only",
            "local-nongraph-reverse",
            "local-nongraph-formula-only-reverse",
            "local-nongraph-formula-plus-mod-reverse",
        }
        for mode in args.modes
    ):
        if args.bridge_file is None:
            args.bridge_file = default_bridge_file_for_benchmark(args.benchmark_file)
        if args.bridge_file is None or not args.bridge_file.exists():
            raise ValueError(
                "bridge-aware modes require --bridge-file or benchmark-adjacent bridge_herb_corpus.jsonl"
            )
        herb_graph = HerbGraph.load(
            args.bridge_file,
            formula_weight=args.formula_bridge_weight,
            condition_weight=args.condition_bridge_weight,
            syndrome_weight=args.syndrome_bridge_weight,
            use_herb_name_prior=not args.disable_herb_name_prior,
            exact_herb_match_weight=args.exact_herb_match_weight,
            partial_herb_match_weight=args.partial_herb_match_weight,
        )

    runtime: base_eval.LocalRuntime | None = None
    if not args.disable_llm_fallback:
        runtime = base_eval.LocalRuntime(
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
            load_embedding=False,
        )

    config = {
        "benchmark_file": str(args.benchmark_file),
        "vocab_file": str(args.vocab_file),
        "case_file": str(args.case_file) if args.case_file else None,
        "bridge_file": str(args.bridge_file) if args.bridge_file else None,
        "modes": args.modes,
        "num_questions": len(benchmark),
        "response_type": args.response_type,
        "llm_backend": args.llm_backend,
        "llm_dir": str(args.llm_dir),
        "llm_device_map": args.llm_device_map,
        "llm_dtype": args.llm_dtype,
        "codex_model": args.codex_model if args.llm_backend == "codex-exec" else None,
        "codex_reasoning_effort": args.codex_reasoning_effort if args.llm_backend == "codex-exec" else None,
        "codex_workdir": str(args.codex_workdir) if args.llm_backend == "codex-exec" else None,
        "codex_audit_dir": str(args.codex_audit_dir) if args.llm_backend == "codex-exec" else None,
        "codex_strict_mode": args.codex_strict_mode if args.llm_backend == "codex-exec" else None,
        "disable_llm_fallback": args.disable_llm_fallback,
        "formula_bridge_weight": args.formula_bridge_weight,
        "condition_bridge_weight": args.condition_bridge_weight,
        "syndrome_bridge_weight": args.syndrome_bridge_weight,
        "disable_herb_name_prior": args.disable_herb_name_prior,
        "exact_herb_match_weight": args.exact_herb_match_weight,
        "partial_herb_match_weight": args.partial_herb_match_weight,
        "force_bridge_generation": args.force_bridge_generation,
        "time_started": int(time.time()),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_summaries: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        print(f"\n=== Running mode={mode} on {len(benchmark)} questions ===", flush=True)
        mode_rows: list[dict[str, Any]] = []
        for index, question in enumerate(benchmark, start=1):
            started = time.time()
            if mode == "llm-only":
                if runtime is None:
                    raise ValueError("mode=llm-only requires the LLM runtime.")
                answer = base_eval.direct_answer_without_retrieval(
                    runtime,
                    question["question"],
                    response_type=args.response_type,
                    max_new_tokens=args.structured_max_tokens,
                )
                context = None
                router_debug = None
                raw_structured_response = None
            elif mode == "bipharm-diag":
                case_router = case_routers.get("full")
                if case_router is None:
                    raise ValueError("mode=bipharm-diag requires cases.")
                answer, context, router_debug, raw_structured_response = run_bipharm_diag(
                    runtime,
                    question,
                    case_router,
                    max_new_tokens=args.structured_max_tokens,
                )
            elif mode == "bipharm-diag-wo-typed-memory":
                case_router = case_routers.get("wo_typed_memory")
                if case_router is None:
                    raise ValueError("mode=bipharm-diag-wo-typed-memory requires cases.")
                answer, context, router_debug, raw_structured_response = run_bipharm_diag(
                    runtime,
                    question,
                    case_router,
                    max_new_tokens=args.structured_max_tokens,
                )
            elif mode == "bipharm-diag-wo-slot-realization":
                case_router = case_routers.get("full")
                if case_router is None:
                    raise ValueError("mode=bipharm-diag-wo-slot-realization requires cases.")
                if runtime is None:
                    raise ValueError("mode=bipharm-diag-wo-slot-realization requires the LLM runtime.")
                answer, context, router_debug, raw_structured_response = run_bipharm_diag(
                    runtime,
                    question,
                    case_router,
                    max_new_tokens=args.structured_max_tokens,
                    force_generation=True,
                )
            elif mode == "bipharm-diag-wo-modification-routing":
                case_router = case_routers.get("wo_modification_routing")
                if case_router is None:
                    raise ValueError("mode=bipharm-diag-wo-modification-routing requires cases.")
                answer, context, router_debug, raw_structured_response = run_bipharm_diag(
                    runtime,
                    question,
                    case_router,
                    max_new_tokens=args.structured_max_tokens,
                )
            elif mode == "bipharm-hg":
                case_router = case_routers.get("full")
                if case_router is None or herb_graph is None:
                    raise ValueError("mode=bipharm-hg requires cases and herb graph.")
                answer, context, router_debug, raw_structured_response = run_bipharm_hg(
                    runtime,
                    question,
                    case_router,
                    herb_graph,
                    max_new_tokens=args.structured_max_tokens,
                    force_bridge_generation=args.force_bridge_generation,
                )
            elif mode == "pharm-lookup":
                if herb_graph is None:
                    raise ValueError("mode=pharm-lookup requires a herb graph.")
                answer, context, router_debug, raw_structured_response = run_pharm_lookup(
                    question,
                    herb_graph,
                )
            elif mode == "bipharm-hg-reverse":
                case_router = case_routers.get("full")
                if case_router is None or herb_graph is None:
                    raise ValueError("mode=bipharm-hg-reverse requires cases and herb graph.")
                answer, context, router_debug, raw_structured_response = run_bipharm_hg_reverse(
                    question,
                    case_router,
                    herb_graph,
                )
            elif mode == "local-nongraph-reverse":
                case_router = case_routers.get("full")
                if case_router is None or herb_graph is None:
                    raise ValueError("mode=local-nongraph-reverse requires cases and herb graph.")
                answer, context, router_debug, raw_structured_response = run_local_nongraph_reverse(
                    question,
                    case_router,
                    herb_graph,
                )
            elif mode == "local-nongraph-formula-only-reverse":
                case_router = case_routers.get("full")
                if case_router is None or herb_graph is None:
                    raise ValueError("mode=local-nongraph-formula-only-reverse requires cases and herb graph.")
                answer, context, router_debug, raw_structured_response = run_local_nongraph_reverse(
                    question,
                    case_router,
                    herb_graph,
                    candidate_strategy="formula_only",
                )
            elif mode == "local-nongraph-formula-plus-mod-reverse":
                case_router = case_routers.get("full")
                if case_router is None or herb_graph is None:
                    raise ValueError("mode=local-nongraph-formula-plus-mod-reverse requires cases and herb graph.")
                answer, context, router_debug, raw_structured_response = run_local_nongraph_reverse(
                    question,
                    case_router,
                    herb_graph,
                    candidate_strategy="formula_plus_modification",
                )
            elif mode == "pharm-clue-only":
                if herb_graph is None:
                    raise ValueError("mode=pharm-clue-only requires a herb graph.")
                answer, context, router_debug, raw_structured_response = run_pharm_clue_only(
                    question,
                    herb_graph,
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            metrics = base_eval.score_question(question, answer, vocab, context=context)
            row = {
                "mode": mode,
                "question_id": question["question_id"],
                "case_id": question.get("case_id"),
                "question_family": question["question_family"],
                "question_type": question["question_type"],
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "gold_answer_aliases": question["gold_answer_aliases"],
                "gold_slots": question["gold_slots"],
                "answer": answer,
                "context": context,
                "answer_strategy": mode,
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
                partial_summary = base_eval.build_summary(mode_rows)
                base_eval.write_jsonl(args.output_dir / f"{mode}_answers.jsonl", mode_rows)
                (args.output_dir / f"{mode}_summary.json").write_text(
                    json.dumps(partial_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        summary = base_eval.build_summary(mode_rows)
        all_summaries[mode] = summary
        base_eval.write_jsonl(args.output_dir / f"{mode}_answers.jsonl", mode_rows)
        (args.output_dir / f"{mode}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        base_eval.write_summary_markdown(args.output_dir / f"{mode}_summary.md", mode, summary)
        base_eval.write_examples_markdown(args.output_dir / f"{mode}_qualitative.md", mode_rows)

    (args.output_dir / "overall_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    base_eval.write_overall_csv(args.output_dir / "overall_summary.csv", all_summaries)
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
