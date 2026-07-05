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

## Command Adapters

The bagging canary can now load a JSON config that registers command-backed
adapters:

```bash
newsbag bagging-canary \
  --manifest /path/to/source_artifacts.jsonl \
  --run-dir /tmp/newsbag_external_smoke \
  --profile external_smoke \
  --config configs/bagging.command.example.json
```

Each command receives formatted arguments such as `{python}`, `{page_id}`,
`{image_path}`, `{profile_path}`, `{output_path}`, `{run_dir}`, `{repo_root}`,
`{model_id}`, `{width}`, and `{height}`. The command must write JSON to
`{output_path}` or print JSON to stdout.

Minimal adapter output:

```json
{
  "page_id": "page-001",
  "model_id": "external_layout_example_v1",
  "regions": [
    {
      "bbox_xyxy": [20, 30, 220, 90],
      "label": "text",
      "confidence": 0.91,
      "text": "recognized text",
      "reading_order": 1
    }
  ]
}
```

This is the intended bridge for real model work: Paddle, Dell, MinerU, or a
local VLM/OCR runner can be wrapped as command adapters first, then promoted to
native Python adapters only if that removes real operational friction.

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
bash scripts/submit_torch_bagging_canary.sh \
  --profile command_fixture \
  --config configs/bagging.command.fixture.json
```
