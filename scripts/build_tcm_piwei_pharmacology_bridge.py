#!/usr/bin/env python3
"""Build bridge records between piwei pathology cases and pharmacopoeia herb entries."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "data" / "benchmarks" / "piwei_v1"
DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "prepared"


LATEX_BLOCK_PATTERN = re.compile(r"\$.*?\$", re.S)
NON_CHINESE_PATTERN = re.compile(r"[A-Za-z0-9_`~^=+|<>/\\\\{}\\[\\]().%-]+")
HERB_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{1,10}")
FORMULA_NAME_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,24}(?:汤|散|丸|饮|方|膏|丹|剂|颗粒|片|胶囊|合剂|口服液|滴丸|软胶囊)")
STRIP_PREFIXES = [
    "麸炒",
    "酒炒",
    "醋炒",
    "盐炒",
    "姜制",
    "酒制",
    "清",
    "法",
    "姜",
    "酒",
    "醋",
    "盐",
    "蜜",
    "炒",
    "炙",
    "生",
    "煅",
    "焦",
    "炮",
]
STRIP_SUFFIXES = [
    "包煎",
    "先煎",
    "后下",
    "冲服",
    "另煎",
    "烊化",
    "研末",
    "外敷",
]
STOPWORDS = {
    "加减",
    "方药",
    "和胃",
    "止痛",
    "止呕",
    "通便",
    "化痰",
    "降逆",
    "温胆汤",
    "左金丸",
    "保和丸",
    "四逆散",
    "半夏厚朴汤",
    "苓桂术甘汤",
}
LEADING_STRIP_CANDIDATES = [
    "加",
    "可加",
    "宜加",
    "少佐",
]
DROP_PREFIXES = ["或用"]
DROP_EXACT_TOKENS = {
    "包煎",
    "先煎",
    "后下",
    "冲服",
    "另煎",
    "外敷",
}
SPECIAL_ALIAS_MAP = {
    "三七粉": ["三七"],
    "代赭石": ["赭石"],
    "积壳": ["枳壳"],
    "蜀椒": ["花椒"],
    "紫苏": ["紫苏叶"],
    "延胡索": ["延胡索(元胡)"],
    "半夏曲": ["半夏"],
    "炮姜炭": ["炮姜"],
    "杏仁": ["苦杏仁"],
    "肉众蓉": ["肉从蓉"],
}


def clean_text(text: str) -> str:
    text = str(text or "").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_name(text: str) -> str:
    return re.sub(r"\s+", "", clean_text(text))


def should_skip_token(name: str) -> bool:
    name = clean_name(name)
    if len(name) < 2 or name in STOPWORDS or name in DROP_EXACT_TOKENS:
        return True
    if any(name.startswith(prefix) for prefix in DROP_PREFIXES):
        return True
    if name.endswith("者"):
        return True
    return bool(FORMULA_NAME_PATTERN.fullmatch(name))


def normalize_herb_name(name: str) -> list[str]:
    name = clean_name(name)
    candidates: list[str] = []
    queue = [name]
    seen: set[str] = set()

    while queue:
        candidate = clean_name(queue.pop(0))
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)

        for mapped in SPECIAL_ALIAS_MAP.get(candidate, []):
            queue.append(mapped)
        for prefix in LEADING_STRIP_CANDIDATES:
            if candidate.startswith(prefix) and len(candidate) > len(prefix) + 1:
                queue.append(candidate[len(prefix):])
        for suffix in STRIP_SUFFIXES:
            if candidate.endswith(suffix) and len(candidate) > len(suffix) + 1:
                queue.append(candidate[: -len(suffix)])
        for prefix in STRIP_PREFIXES:
            if candidate.startswith(prefix) and len(candidate) > len(prefix) + 1:
                queue.append(candidate[len(prefix):])

    deduped: list[str] = []
    for candidate in candidates:
        candidate = clean_name(candidate)
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def extract_formula_herbs(formula_text: str) -> list[str]:
    formula_text = clean_text(formula_text)
    if "。" in formula_text:
        formula_text = formula_text.split("。", 1)[1]
    formula_text = LATEX_BLOCK_PATTERN.sub(" ", formula_text)
    formula_text = NON_CHINESE_PATTERN.sub(" ", formula_text)
    formula_text = FORMULA_NAME_PATTERN.sub(" ", formula_text)
    herbs: list[str] = []
    seen: set[str] = set()
    for token in HERB_TOKEN_PATTERN.findall(formula_text):
        token = clean_name(token)
        if should_skip_token(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        herbs.append(token)
    return herbs


def build_pharm_index(herb_table: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in herb_table:
        for alias in row.get("aliases", []) + [row.get("herb_name", "")]:
            alias = clean_name(alias)
            if alias:
                index[alias].append(row)
    return index


def choose_best_match(name: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    normalized_candidates = normalize_herb_name(name)
    for target in normalized_candidates:
        for row in candidates:
            if clean_name(row.get("herb_name", "")) == target:
                return row
    for target in normalized_candidates:
        for row in candidates:
            if target in (clean_name(alias) for alias in row.get("aliases", [])):
                return row
    return candidates[0]


def match_pharm_herb(
    name: str,
    pharm_index: dict[str, list[dict[str, Any]]],
) -> tuple[str, list[str], dict[str, Any] | None, str | None]:
    herb = clean_name(name)
    if not herb or should_skip_token(herb):
        return herb, [], None, None

    aliases = [alias for alias in normalize_herb_name(herb) if not should_skip_token(alias)]
    matched = None
    matched_alias = None
    for alias in aliases:
        candidates = pharm_index.get(alias, [])
        matched = choose_best_match(herb, candidates)
        if matched is not None:
            matched_alias = alias
            break
    return herb, aliases, matched, matched_alias


def build_bridge_records(
    cases: list[dict[str, Any]],
    herb_table: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pharm_index = build_pharm_index(herb_table)
    rows: list[dict[str, Any]] = []
    match_type_counter = Counter()
    bridge_type_counter = Counter()
    unmatched_counter = Counter()
    unique_formula_herbs: set[str] = set()
    unique_mod_herbs: set[str] = set()
    skipped_formula_tokens = 0
    skipped_mod_tokens = 0

    for case in cases:
        formula_herbs = extract_formula_herbs(case.get("formula_text", ""))
        for herb in formula_herbs:
            herb, aliases, matched, matched_alias = match_pharm_herb(herb, pharm_index)
            if not herb or not aliases:
                skipped_formula_tokens += 1
                continue
            unique_formula_herbs.add(herb)
            if matched is None:
                unmatched_counter[herb] += 1
            else:
                match_type_counter["matched"] += 1
            bridge_type_counter["formula_herb"] += 1
            rows.append(
                {
                    "bridge_type": "formula_herb",
                    "case_id": case.get("case_id"),
                    "syndrome": case.get("syndrome"),
                    "pathogenesis": case.get("pathogenesis"),
                    "therapy": case.get("therapy"),
                    "formula_name": case.get("formula_name"),
                    "condition": None,
                    "herb_name": herb,
                    "normalized_aliases": aliases,
                    "matched_pharm_herb": matched.get("herb_name") if matched else None,
                    "matched_alias": matched_alias,
                    "pharm_efficacy": matched.get("efficacy") if matched else "",
                    "pharm_indications_text": matched.get("indications_text") if matched else "",
                    "pharm_indication_terms": matched.get("indication_terms") if matched else [],
                    "pharm_dosage": matched.get("dosage") if matched else "",
                    "pharm_course": matched.get("course") if matched else "",
                    "pharm_caution_text": matched.get("caution_text") if matched else "",
                    "pharm_source_record_id": matched.get("source_record_id") if matched else None,
                }
            )

        for mod in case.get("modification_pairs", []):
            for herb in mod.get("herbs", []):
                herb, aliases, matched, matched_alias = match_pharm_herb(herb, pharm_index)
                if not herb or not aliases:
                    skipped_mod_tokens += 1
                    continue
                unique_mod_herbs.add(herb)
                if matched is None:
                    unmatched_counter[herb] += 1
                else:
                    match_type_counter["matched"] += 1
                bridge_type_counter["modification_herb"] += 1
                rows.append(
                    {
                        "bridge_type": "modification_herb",
                        "case_id": case.get("case_id"),
                        "syndrome": case.get("syndrome"),
                        "pathogenesis": case.get("pathogenesis"),
                        "therapy": case.get("therapy"),
                        "formula_name": case.get("formula_name"),
                        "condition": mod.get("condition"),
                        "herb_name": herb,
                        "normalized_aliases": aliases,
                        "matched_pharm_herb": matched.get("herb_name") if matched else None,
                        "matched_alias": matched_alias,
                        "pharm_efficacy": matched.get("efficacy") if matched else "",
                        "pharm_indications_text": matched.get("indications_text") if matched else "",
                        "pharm_indication_terms": matched.get("indication_terms") if matched else [],
                        "pharm_dosage": matched.get("dosage") if matched else "",
                        "pharm_course": matched.get("course") if matched else "",
                        "pharm_caution_text": matched.get("caution_text") if matched else "",
                        "pharm_source_record_id": matched.get("source_record_id") if matched else None,
                    }
                )

    total_rows = len(rows)
    matched_rows = sum(1 for row in rows if row["matched_pharm_herb"])
    matched_mod_rows = sum(
        1 for row in rows if row["bridge_type"] == "modification_herb" and row["matched_pharm_herb"]
    )
    matched_formula_rows = sum(
        1 for row in rows if row["bridge_type"] == "formula_herb" and row["matched_pharm_herb"]
    )
    stats = {
        "num_cases": len(cases),
        "num_bridge_rows": total_rows,
        "bridge_type_counts": dict(sorted(bridge_type_counter.items())),
        "matched_rows": matched_rows,
        "matched_ratio": round(matched_rows / total_rows, 4) if total_rows else 0.0,
        "matched_formula_rows": matched_formula_rows,
        "matched_modification_rows": matched_mod_rows,
        "unique_formula_herbs": len(unique_formula_herbs),
        "unique_modification_herbs": len(unique_mod_herbs),
        "skipped_formula_tokens": skipped_formula_tokens,
        "skipped_modification_tokens": skipped_mod_tokens,
        "top_unmatched_herbs": unmatched_counter.most_common(30),
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-file",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR / "cases.jsonl",
    )
    parser.add_argument(
        "--herb-table-file",
        type=Path,
        default=DEFAULT_PREPARED_DIR / "pharmacopoeia_herb_table.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_PREPARED_DIR / "piwei_pharmacology_bridge.jsonl",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=DEFAULT_PREPARED_DIR / "piwei_pharmacology_bridge_stats.json",
    )
    args = parser.parse_args()

    cases = load_jsonl(args.case_file)
    herb_table = load_jsonl(args.herb_table_file)
    bridge_rows, stats = build_bridge_records(cases, herb_table)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_file, bridge_rows)
    args.stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
