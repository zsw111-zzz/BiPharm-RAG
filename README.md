# BiPharm-RAG

Code for **BiPharm-RAG: Cross-source dual-hypergraph retrieval augmented large language models for traditional Chinese medicine diagnosis and treatment reasoning**.

This repository contains the core runtime and evaluation scripts used for the main experiments reported in the paper:

- pathology-side evaluation on the disease-monograph benchmark
- reverse bridge evaluation from pathology clues to pharmacopoeia herb targets
- result aggregation across books

## Repository layout

- `bipharm_rag/`
  Core BiPharm-RAG runtime.
- `scripts/run_tcm_piwei_eval.py`
  Main pathology benchmark runner for direct LLM, naive retrieval, and Cog-RAG-family baselines under a shared evaluation pipeline.
- `scripts/run_tcm_bipharm_eval.py`
  BiPharm-RAG runner for the pathology benchmark and reverse bridge benchmark.
- `scripts/build_tcm_piwei_pharmacology_bridge.py`
  Utility for constructing bridge records between pathology cases and pharmacopoeia herb entries.
- `scripts/summarize_tcm_multibook_results.py`
  Utility for aggregating per-book outputs into summary tables.
- `data/README.md`
  Expected local data layout. The copyrighted source texts and source-derived benchmark records are not bundled here.

## Environment

Tested with Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected local layout

The repository assumes the following optional local directories:

```text
BiPharm-RAG/
├── bipharm_rag/
├── scripts/
├── data/
│   ├── benchmarks/
│   └── prepared/
└── models/
```

- `data/benchmarks/` stores benchmark files such as `qa_benchmark.jsonl`, `vocab.json`, and `cases.jsonl`.
- Retrieval-backed pathology evaluation also expects a source-corpus file such as `corpus.jsonl`.
- `data/prepared/` stores prepared pharmacopoeia-side tables or bridge files.
- `models/` stores local model checkpoints if you run with `--llm-backend local`.

If your local paths differ, pass them explicitly via CLI arguments.

For retrieval-backed pathology evaluation, the first run must either:

- pass `--data-file` so the script can build the retrieval cache in `--working-dir`, or
- reuse an already populated `--working-dir`.

When `--data-file` is a `.jsonl` file, each line should contain a `text` field and may optionally contain `section_path`.

## Example commands

Pathology benchmark, first run with cache construction:

```bash
python scripts/run_tcm_piwei_eval.py \
  --benchmark-file data/benchmarks/piwei_v1/qa_benchmark.jsonl \
  --vocab-file data/benchmarks/piwei_v1/vocab.json \
  --case-file data/benchmarks/piwei_v1/cases.jsonl \
  --data-file data/benchmarks/piwei_v1/corpus.jsonl \
  --working-dir outputs/piwei_work \
  --output-dir outputs/piwei_eval \
  --llm-dir models/Qwen/Qwen3.5-2B \
  --emb-dir models/BAAI/bge-m3 \
  --modes llm-only naive cog cog-hybrid cog-entity cog-theme \
  --answer-strategy direct
```

Pathology benchmark, reusing an existing cache:

```bash
python scripts/run_tcm_piwei_eval.py \
  --benchmark-file data/benchmarks/piwei_v1/qa_benchmark.jsonl \
  --vocab-file data/benchmarks/piwei_v1/vocab.json \
  --case-file data/benchmarks/piwei_v1/cases.jsonl \
  --working-dir outputs/piwei_work \
  --output-dir outputs/piwei_eval \
  --llm-dir models/Qwen/Qwen3.5-2B \
  --emb-dir models/BAAI/bge-m3 \
  --modes llm-only naive cog cog-hybrid cog-entity cog-theme \
  --answer-strategy direct
```

BiPharm-RAG evaluation:

```bash
python scripts/run_tcm_bipharm_eval.py \
  --benchmark-file data/benchmarks/piwei_v1/qa_benchmark.jsonl \
  --vocab-file data/benchmarks/piwei_v1/vocab.json \
  --case-file data/benchmarks/piwei_v1/cases.jsonl \
  --bridge-file data/prepared/piwei_pharmacology_bridge.jsonl \
  --output-dir outputs/bipharm_piwei \
  --llm-dir models/Qwen/Qwen3.5-4B \
  --modes llm-only bipharm-diag
```

Bridge construction:

```bash
python scripts/build_tcm_piwei_pharmacology_bridge.py \
  --case-file data/benchmarks/piwei_v1/cases.jsonl \
  --herb-table-file data/prepared/pharmacopoeia_herb_table.jsonl \
  --output-file data/prepared/piwei_pharmacology_bridge.jsonl \
  --stats-file data/prepared/piwei_pharmacology_bridge_stats.json
```

Result aggregation:

```bash
python scripts/summarize_tcm_multibook_results.py \
  --result-dirs outputs/piwei_eval outputs/bipharm_piwei \
  --output-csv outputs/summary/all_results.csv \
  --output-md outputs/summary/all_results.md
```

## Data note

The study uses OCR-normalized disease monographs and Chinese Pharmacopoeia text obtained from third-party sources. Because the underlying texts and many source-derived benchmark records are subject to copyright and redistribution restrictions, they are not bundled in this repository.
