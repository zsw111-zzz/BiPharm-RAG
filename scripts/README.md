# Scripts

- `run_tcm_piwei_eval.py`
  Runs the pathology benchmark under a shared pipeline for direct LLM, naive retrieval, and Cog-RAG-family baselines.
- `run_tcm_bipharm_eval.py`
  Runs BiPharm-RAG on pathology and reverse bridge benchmarks.
- `build_tcm_piwei_pharmacology_bridge.py`
  Builds bridge records linking pathology-side cases to pharmacopoeia-side herb entries.
- `summarize_tcm_multibook_results.py`
  Aggregates per-book outputs into summary tables.

All dataset and model paths can be passed explicitly from the command line.
