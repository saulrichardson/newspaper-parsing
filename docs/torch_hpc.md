# Torch HPC Operations

This pipeline is optimized for NYU Torch with GPU-first execution and clear CPU fallback.

## Account / Partition / GRES

Use valid combinations only:

- `-A torch_pr_609_general -p l40s_public --gres=gpu:l40s:1`
- `-A torch_pr_609_general -p h200_public --gres=gpu:h200:1`

Always use `--gres=...` on Torch. Do not use `--gpus=1`.

## Preflight

```bash
sbatch --test-only -A torch_pr_609_general -p l40s_public \
  --gres=gpu:l40s:1 --cpus-per-task=4 --mem=16G --time=00:05:00 --wrap hostname
```

## Environment Management

This repo intentionally separates environments (and keeps them decoupled):

- `newsbag` orchestration env (lightweight: `numpy`, `Pillow`)
- PaddleOCR CLI env (Paddle layout + VL1.5 doc_parser)
- Dell env (ONNXRuntime CUDA)
- MinerU env (Transformers + `mineru_vl_utils`)

Default in shipped Slurm scripts:
- `VENV_DIR` defaults to `$BASE/envs/mineru25_py310` because Torch commonly has this env available
  with Python 3.10+ (compatible with `newsbag`).
- You can override by exporting `VENV_DIR` or `NEWSBAG_PY`.

### Recommended orchestration env

Use a scratch-backed venv for the orchestration layer (this repo). It stays lightweight and
does not need the heavy model deps because the pipeline calls model runners via subprocess.

```bash
BASE=/scratch/$USER/paddleocr_vl15
PROJECT_ROOT="$BASE/newspaper-parsing"
python3 -m venv $BASE/envs/newsbag
source $BASE/envs/newsbag/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "$PROJECT_ROOT"
```

### Optional fallback: avoid venv Python mismatch

If you ever hit a Torch node mismatch where the `venv` python no longer runs on compute nodes,
you can run the orchestration with a portable Conda env python instead:

```bash
BASE=/scratch/$USER/paddleocr_vl15
PROJECT_ROOT="$BASE/newspaper-parsing"
export NEWSBAG_PY="$BASE/envs/mineru25_py310/bin/python"
$NEWSBAG_PY -m pip install -e "$PROJECT_ROOT"
```

Model envs are referenced by path in `configs/pipeline.torch.json`. Verify they exist:

```bash
BASE=/scratch/$USER/paddleocr_vl15
$BASE/envs/paddleocr_gpu_py312/bin/paddleocr --help
$BASE/envs/mineru25_py310/bin/python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
$BASE/envs/mineru25_py310/bin/python -c 'import mineru_vl_utils; print(mineru_vl_utils.__version__)'
```

## GPU Utilization Guardrails

Torch can cancel jobs with low average GPU utilization. To avoid that:

- Keep GPU-heavy inference stages (Paddle, Dell, MinerU) in GPU jobs.
- Run fusion + review PNG generation in a CPU job (do not idle GPUs).
- If Dell resolves to CPU provider (`CPUExecutionProvider`), do not run it on a GPU partition.
- Use short canary jobs before long stress runs.

## Runtime Checks

- `nvidia-smi` at job start.
- Check `outputs/sources/dell/<variant>/run_report.json` for `providers_used`.
- Check `outputs/sources/mineru/<variant>/run_meta.json` for `cuda_available`.

## Suggested Submission Flow

Do not run fusion/review inside a GPU allocation: it will idle the GPU and can trip low-util cancellation.

Recommended (two-job chain):

1. GPU inference-only job: `paddle_layout,paddle_vl15,dell,mineru`
2. CPU post-processing job: `fusion,review`

### Split GPU Mode (Recommended When Using H200)
As of February 2026, Paddle stages can fail on Torch `h200_public` with CUDA kernel image mismatch errors.
To keep throughput high while still using H200, run a split GPU flow:

1. L40S GPU job: `paddle_layout,paddle_vl15`
2. H200 GPU job: `dell,mineru`
3. CPU job: `fusion,review` (afterok on both GPU jobs)

One-command helper (run on Torch login, from inside the repo):

```bash
cd /scratch/$USER/paddleocr_vl15/newspaper-parsing
bash torch/slurm/submit_newsbag_full.sh
```

End-to-end helper when you have a directory of scans:

```bash
cd /scratch/$USER/paddleocr_vl15/newspaper-parsing
bash torch/slurm/submit_newsbag_from_dir.sh \
  --input-dir /scratch/$USER/paddleocr_vl15/input/ad_hoc_newspapers_20260205_190618 \
  --recursive \
  --gpu l40s
```

Split GPU helper:

```bash
cd /scratch/$USER/paddleocr_vl15/newspaper-parsing
bash torch/slurm/submit_newsbag_from_dir.sh \
  --input-dir /scratch/$USER/paddleocr_vl15/input/stress_iter10_pages \
  --recursive \
  --gpu split
```

Manual submission (explicit run dir + dependency):

```bash
BASE=/scratch/$USER/paddleocr_vl15
RUN_DIR="$BASE/runs/layout_bagging_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# Optional override if your venv python breaks on a compute node:
# export NEWSBAG_PY="$BASE/envs/mineru25_py310/bin/python"

JID_INFER="$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR" torch/slurm/newsbag_infer_l40s.sbatch)"
sbatch --dependency=afterok:$JID_INFER --export=ALL,RUN_DIR="$RUN_DIR" torch/slurm/newsbag_fuse_review_cs.sbatch
```

Review:
- `outputs/fusion/variant_leaderboard.tsv`
- `outputs/fusion/source_leaderboard.tsv`
- `review/top20_informative/pages/`
