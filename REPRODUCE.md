# Reproduction notes

This repository contains the released BiPharm-RAG runtime and evaluation scripts. Full benchmark reruns require locally available benchmark files, prepared pharmacopoeia tables, and accessible model checkpoints or API backends.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal checks

```bash
python scripts/run_tcm_piwei_eval.py --help
python scripts/run_tcm_bipharm_eval.py --help
python scripts/build_tcm_piwei_pharmacology_bridge.py --help
python scripts/summarize_tcm_multibook_results.py --help
```

## Full reruns

For full reruns, place benchmark and prepared files under `data/` or pass their locations explicitly by command line.
