# Data layout

This repository does not bundle the copyrighted source texts or the full source-derived benchmark files used in the paper.

Expected local layout:

```text
data/
├── benchmarks/
│   └── <benchmark_name>/
│       ├── qa_benchmark.jsonl
│       ├── vocab.json
│       ├── cases.jsonl
│       ├── corpus.jsonl
│       └── stats.json
└── prepared/
    ├── pharmacopoeia_herb_table.jsonl
    ├── <bridge_name>.jsonl
    └── <bridge_name>_stats.json
```

`corpus.jsonl` is used to build the retrieval cache for pathology-side evaluation. Each JSONL line should contain a `text` field and may optionally contain `section_path`.

If you store these files elsewhere, pass the corresponding paths explicitly to the evaluation scripts.
