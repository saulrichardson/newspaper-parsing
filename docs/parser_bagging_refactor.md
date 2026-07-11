# Parser Bagging Refactor

`newspaper-parsing` is the manifest-driven model-bagging engine for the
newspaper stack.

The new forward-looking contract is:

1. acquisition emits parse input manifests with stable page IDs and source
   provenance;
2. parsing runs configurable model adapters over those pages;
3. each adapter emits normalized model output;
4. fusion emits canonical page layouts, transcripts, quality scores, review
   packets, input validation, performance metrics, and provenance.

The initial `newsbag bagging-canary` command is intentionally lightweight. It
uses deterministic CPU adapters so local tests and Torch Slurm canaries validate
the infrastructure without requiring a full OCR/VLM production run.

## Planning And Adaptive Routing

Model execution starts with a deterministic planning pass:

```bash
newsbag plan-bagging \
  --manifest /path/to/source_artifacts.jsonl \
  --run-dir /tmp/newsbag_plan \
  --profile adaptive \
  --config configs/bagging.command.example.json
```

Planning writes the copied input manifest, one page profile per page,
`manifests/model_plan.jsonl`, and `reports/plan_summary.json`. It does not run
model adapters. Each command adapter may declare `complexities` containing any
of `easy`, `medium`, and `hard`; omitted complexity routing means all three.
The plan therefore states what should execute before CPU/GPU allocation or
model startup.

## Command Adapters

Validate a parse input manifest before running model work:

```bash
newsbag validate-parse-input-manifest \
  --manifest /path/to/source_artifacts.jsonl \
  --require-files \
  --verify-checksums \
  --output-json /tmp/source_artifacts.validation.json
```

The same validation report is written into every canary run under
`reports/input_manifest_validation.json`; `newsbag validate-run` treats an
errored input manifest report as a failed run bundle.

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
`{image_path}`, `{image_dir}`, `{image_stem}`, `{image_name}`,
`{manifest_dir}`, `{profile_path}`, `{output_path}`, `{run_dir}`,
`{repo_root}`, `{model_id}`, `{width}`, and `{height}`. The command must write
JSON to `{output_path}` or print JSON to stdout.

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

## Importing Legacy Model Outputs

Existing Paddle, Dell, and MinerU runs can be brought into the new bagging
contract without rerunning the model. The converter accepts the older normalized
layout shapes (`boxes`, `regions`, `candidates`, nested `res.boxes`, or
page-level `pages`) and emits a `ModelOutput` JSON file:

```bash
python scripts/legacy_layout_to_model_output.py \
  --input-json /path/to/legacy-normalized/page-001.json \
  --page-id page-001 \
  --model-id legacy_paddle_layout_v1 \
  --source-family paddle \
  --output-json /tmp/page-001.model_output.json
```

The same converter can be registered as a command adapter:

```bash
newsbag bagging-canary \
  --manifest /path/to/source_artifacts.jsonl \
  --run-dir /tmp/newsbag_legacy_import \
  --profile legacy_import \
  --config configs/bagging.legacy-output.example.json
```

This is the fastest path for comparing old experiments inside the new fusion
and provenance layer. Live GPU-backed adapters can replace these imports later
without changing the run bundle contract.

For a full old `newsbag run`, generate the adapter config instead of writing it
by hand:

```bash
newsbag legacy-run-config \
  --legacy-run-dir /path/to/old/run \
  --output-config /tmp/legacy-bagging.json \
  --output-summary /tmp/legacy-bagging.summary.json \
  --profile legacy_import
```

The generated adapters scan the legacy source layout:

```text
outputs/sources/{paddle_layout,paddle_vl15,dell,mineru}/<variant>/<page_slug>/layout_boxes.normalized.json
```

Generated adapters use `--allow-missing` by default so a page missing one
legacy model output emits an empty skipped `ModelOutput` instead of failing the
entire page. Use `--strict-missing` if missing source outputs should fail fast.

## Run Validation

Every parser-bagging run can be validated as a run bundle:

```bash
newsbag validate-run \
  --run-dir /tmp/newsbag_bagging_canary \
  --output-json /tmp/newsbag_bagging_canary/reports/validation.json
```

The validator checks the copied parse manifest, page-level model plan,
`reports/input_manifest_validation.json`, `summary.json`, `provenance.json`,
raw and aggregate performance reports, `errors.jsonl`, every successful model
output, fused page JSON, transcript file, region bounding boxes, per-page model
IDs, plan coverage, and headline counts. The Torch Slurm canary writes this validation report into
`reports/validation.json` and includes the validation status in
`slurm_status.json`.

One failed adapter is recorded with `scope=adapter` and does not invalidate
successful model outputs for that page. The page can still be fused from the
remaining outputs, while the overall run remains visibly errored. A page-level
failure uses `scope=page` and is excluded from completed-page counts.

## Performance Reports

Every run writes raw orchestration timings to `reports/performance.jsonl` and
an aggregate `reports/performance_summary.json`. Aggregates include stage,
model, resource-class, and page-complexity distributions; p50/p95/max latency;
serialized throughput; slowest rows; and planned-versus-observed invocation
coverage.

```bash
newsbag summarize-performance --run-dir /tmp/newsbag_bagging_canary
```

The command rebuilds the aggregate from the raw rows, which makes performance
reporting reproducible and independently testable.

## Torch Canary

From the local repo:

```bash
bash scripts/submit_torch_bagging_canary.sh
```

The submitter syncs this repo to `/scratch/$USER/codex_hpc/parser_bagging`,
creates or reuses a lightweight `newsbag` virtualenv, generates a tiny fixture
manifest when needed, submits
`torch/slurm/newsbag_bagging_canary_cpu_short.sbatch`,
waits for completion by default, and prints `slurm_status.json`.

Useful variants:

```bash
bash scripts/submit_torch_bagging_canary.sh --profile baseline --plan-only
bash scripts/submit_torch_bagging_canary.sh --profile adaptive --no-wait
bash scripts/submit_torch_bagging_canary.sh --skip-sync --timeout 1200
bash scripts/submit_torch_bagging_canary.sh \
  --profile command_fixture \
  --config configs/bagging.command.fixture.json
bash scripts/submit_torch_bagging_canary.sh \
  --profile legacy_fixture \
  --config configs/bagging.legacy.fixture.json
```
