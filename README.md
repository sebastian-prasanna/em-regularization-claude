# Implicit regularization and emergent misalignment

Does weight decay, batch size or LoRA rank change emergent misalignment (EM) at matched in-domain loss?
Qwen3-8B and Qwen3.8-27B, LoRA via Tinker, Turner/Soligo narrow-misalignment datasets.

- Write-up: [`blog/blogpost.md`](blog/blogpost.md)
- Per-checkpoint results (gpt-4o judge): `results/<run>/summary.gpt-4o-2024-08-06.jsonl`

Datasets are not included (their terms forbid redistribution). Fetch and decrypt them from
https://github.com/clarifying-EM/model-organisms-for-EM, then set `DATA_DIR` in `common.py`.

Pipeline: `train.py` (fine-tune + sample eval generations) → `analyze.py` (judge) → `domain.py`
(off-domain filter) → `frontier.py` / `plot.py` / `summary_fig.py` (analysis and figures).
`launch.py configs_*.jsonl N` runs a config list with N concurrent jobs.
