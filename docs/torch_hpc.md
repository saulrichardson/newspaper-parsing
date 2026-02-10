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

This repo intentionally separates environments:

- `newsbag` orchestration env (lightweight: `numpy`, `Pillow`)
- PaddleOCR CLI env (Paddle layout + VL1.5 doc_parser)
- Dell env (ONNXRuntime CUDA)
- MinerU env (Transformers + `mineru_vl_utils`)

Recommended (scratch-backed venv for orchestration):

```bash
BASE=/scratch/$USER/paddleocr_vl15
python3 -m venv $BASE/envs/newsbag
source $BASE/envs/newsbag/bin/activate
python -m pip install --upgrade pip
python -m pip install -e $BASE/new-ocr
```

Model envs are referenced by path in `configs/pipeline.torch.json`. Verify they exist:

```bash
BASE=/scratch/$USER/paddleocr_vl15
$BASE/envs/paddleocr_gpu_py312/bin/paddleocr --help
$BASE/envs/doclayout_l40s/bin/python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
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

One-command helper (run on Torch login, from inside the repo):

```bash
cd /scratch/$USER/paddleocr_vl15/new-ocr
bash torch/slurm/submit_newsbag_full.sh
```

End-to-end helper when you have a directory of scans:

```bash
cd /scratch/$USER/paddleocr_vl15/new-ocr
bash torch/slurm/submit_newsbag_from_dir.sh \
  --input-dir /scratch/$USER/paddleocr_vl15/input/ad_hoc_newspapers_20260205_190618 \
  --recursive \
  --gpu l40s
```

Manual submission (explicit run dir + dependency):

```bash
BASE=/scratch/$USER/paddleocr_vl15
RUN_DIR="$BASE/runs/layout_bagging_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

JID_INFER="$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR" torch/slurm/newsbag_infer_l40s.sbatch)"
sbatch --dependency=afterok:$JID_INFER --export=ALL,RUN_DIR="$RUN_DIR" torch/slurm/newsbag_fuse_review_cs.sbatch
```

Review:
- `outputs/fusion/variant_leaderboard.tsv`
- `outputs/fusion/source_leaderboard.tsv`
- `review/top20_informative/pages/`
