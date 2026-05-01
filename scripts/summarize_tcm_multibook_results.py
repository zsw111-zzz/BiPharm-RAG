#!/usr/bin/env python3
"""Collect multi-book TCM evaluation results into one CSV and Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_book_name(run_config: dict[str, Any]) -> str:
    benchmark_file = Path(run_config.get("benchmark_file", ""))
    stats_file = benchmark_file.parent / "stats.json"
    if stats_file.exists():
        stats = load_json(stats_file)
        return str(stats.get("book_name", benchmark_file.parent.name))
    return benchmark_file.parent.name


def collect_run_rows(result_dir: Path) -> list[dict[str, Any]]:
    run_config_file = result_dir / "run_config.json"
    if not run_config_file.exists():
        return []
    run_config = load_json(run_config_file)
    book_name = detect_book_name(run_config)
    answer_strategy = str(run_config.get("answer_strategy", "direct"))

    rows: list[dict[str, Any]] = []
    for summary_file in sorted(result_dir.glob("*_summary.json")):
        mode = summary_file.stem.removesuffix("_summary")
        summary = load_json(summary_file)
        overall = summary.get("overall", {})
        rows.append(
            {
                "result_dir": str(result_dir),
                "book_name": book_name,
                "answer_strategy": answer_strategy,
                "mode": mode,
                "num_questions": overall.get("num_questions", 0),
                "target_hit_rate": overall.get("target_hit_rate", 0.0),
                "exact_match_rate": overall.get("exact_match_rate", 0.0),
                "avg_char_f1": overall.get("avg_char_f1", 0.0),
                "avg_slot_precision": overall.get("avg_slot_precision", 0.0),
                "avg_slot_recall": overall.get("avg_slot_recall", 0.0),
                "avg_slot_f1": overall.get("avg_slot_f1", 0.0),
                "all_slots_hit_rate": overall.get("all_slots_hit_rate", 0.0),
                "context_hit_rate": overall.get("context_hit_rate"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "book_name",
        "answer_strategy",
        "mode",
        "num_questions",
        "target_hit_rate",
        "exact_match_rate",
        "avg_char_f1",
        "avg_slot_precision",
        "avg_slot_recall",
        "avg_slot_f1",
        "all_slots_hit_rate",
        "context_hit_rate",
        "result_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Multi-book Result Summary",
        "",
        "| Disease book | Strategy | Mode | N | Hit | Exact | CharF1 | SlotF1 | ContextHit | Result dir |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['book_name']} | {row['answer_strategy']} | {row['mode']} | {row['num_questions']} | "
            f"{fmt_metric(row['target_hit_rate'])} | {fmt_metric(row['exact_match_rate'])} | "
            f"{fmt_metric(row['avg_char_f1'])} | {fmt_metric(row['avg_slot_f1'])} | "
            f"{fmt_metric(row['context_hit_rate'])} | {row['result_dir']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for result_dir in args.result_dirs:
        rows.extend(collect_run_rows(result_dir))
    rows.sort(key=lambda row: (row["book_name"], row["answer_strategy"], row["mode"]))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(json.dumps({"num_rows": len(rows), "output_csv": str(args.output_csv), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
