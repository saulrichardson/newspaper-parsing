# Parser Bagging Refactor

This branch makes `newspaper-parsing` the manifest-driven model-bagging engine
for the newspaper stack.

The new forward-looking contract is:

1. acquisition emits parse input manifests with stable page IDs and source
   provenance;
2. parsing runs configurable model adapters over those pages;
3. each adapter emits normalized model output;
4. fusion emits canonical page layouts, transcripts, quality scores, review
   packets, performance metrics, and provenance.

The initial `newsbag bagging-canary` command is intentionally lightweight. It
uses deterministic CPU adapters so local tests and Torch Slurm canaries validate
the infrastructure without requiring a full OCR/VLM production run.

## Torch Canary

From the local repo:

```bash
bash scripts/submit_torch_bagging_canary.sh
```

The submitter syncs this repo to `/scratch/$USER/codex_hpc/parser_bagging`,
creates or reuses a lightweight `newsbag` virtualenv, generates a tiny fixture
manifest when needed, submits `torch/slurm/newsbag_bagging_canary_cs.sbatch`,
waits for completion by default, and prints `slurm_status.json`.

Useful variants:

```bash
bash scripts/submit_torch_bagging_canary.sh --profile baseline --plan-only
bash scripts/submit_torch_bagging_canary.sh --profile adaptive --no-wait
bash scripts/submit_torch_bagging_canary.sh --skip-sync --timeout 1200
```
