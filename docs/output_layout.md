# Output Layout Contract

This document defines the canonical output structure of one `newsbag run`.

## Run Root

`<run_dir>/`

- `manifests/`
  - `images.resolved.txt`: one absolute image path per line.
  - `config.resolved.json`: fully resolved config snapshot for reproducibility.
- `logs/`: command logs per stage/page.
- `reports/`: stage summaries and scoring tables.
- `outputs/`
  - `sources/`
    - `paddle_layout/<variant>/<slug>/`
      - `<slug>_res.json`: raw Paddle layout output.
      - `<slug>_res.png`: raw Paddle overlay.
      - `layout_boxes.normalized.json`: normalized boxes (`source_label` and `norm_label` kept).
      - `labels_source_counts.json`: raw label histogram.
    - `reports/paddle_layout_<variant>_labels_aggregate.json`: run-level label histograms (raw + normalized).
    - `paddle_vl15/<variant>/<slug>/`
      - `layout_boxes.normalized.json`: VL1.5 layout stream as boxes.
      - `labels_source_counts.json`: VL1.5 layout labels.
      - `parsing_blocks.json`: full `parsing_res_list` semantic blocks.
      - `table_blocks.html`: concatenated table block HTML (if present).
      - `raw/`: raw files emitted by `paddleocr doc_parser`.
    - `reports/paddle_vl15_<variant>_labels_aggregate.json`: run-level label + block histograms.
    - `dell/<variant>/<slug>/`
      - `<slug>_dell_layout_boxes.json`: raw Dell boxes.
      - `<slug>_dell_layout_overlay.png`: raw Dell overlay.
      - `layout_boxes.normalized.json`: normalized boxes.
      - `labels_source_counts.json`: raw label histogram.
    - `reports/dell_<variant>_labels_aggregate.json`: run-level label histograms + providers_used.
    - `mineru/<variant>/<slug>/`
      - `<slug>_mineru_layout_boxes.json`: raw MinerU boxes.
      - `<slug>_mineru_overlay.png`: raw MinerU overlay.
      - `<slug>_mineru_raw.json`: raw MinerU extraction payload.
      - `layout_boxes.normalized.json`: normalized boxes.
      - `labels_source_counts.json`: raw label histogram.
    - `reports/mineru_<variant>_labels_aggregate.json`: run-level label histograms + cuda_available.
  - `fusion/`
    - `<variant>/<slug>/`
      - `fused_boxes.json`: fused candidate blocks.
      - `metrics.json`: per-page metrics for that variant.
      - `fused_layout.png`: fused overlay (recommended variant only).
      - `consensus_coverage_overlay.png`: consensus proxy overlay (recommended variant only).
    - `summary.json`: all pages, all variants, leaderboard, recommended variant.
    - `variant_leaderboard.tsv`: variant ranking table.
    - `per_page_variant_metrics.tsv`: page-level metric rows.
    - `source_leaderboard.tsv`: per-parser ranking table (each Paddle variant + Dell + MinerU).
    - `per_page_source_metrics.tsv`: page-level metric rows per parser.
  - `transcription/<variant>/`
    - `transcription_report.tsv`: per-page status + counts (`ocr_line_count`, `assigned_line_count`, etc.).
    - `transcript_combined.txt`: all pages concatenated in heading order.
    - `<slug>/`
      - `ocr.log`: OCR command log for the page.
      - `ocr_raw.json`: ROI OCR metadata (`mode=roi_fused`, crop list, crop-level raw paths and line counts).
      - `ocr_lines.json`: parsed OCR lines remapped to page coordinates (`mode=roi_fused`).
      - `transcript_boxes.json`: fused boxes with attached OCR lines/text (one OCR assignment bin per fused box).
      - `transcript.txt`: page transcript (ordered fused text/title regions).
      - `ocr_regions_overlay.png`: original page with cleaned OCR region boxes used for crop OCR.
      - `ocr_lines_overlay.png`: original page with remapped OCR line boxes.
      - `ocr_raw_dir/`: Paddle OCR artifacts for the fused crops.
        - `crops/crop_XXXX.png`: input crops cut from fused boxes.
        - `crop_XXXX_res.json`, `crop_XXXX_ocr_res_img.png`, ...: per-crop OCR outputs.
- `review/`
  - `pages/<slug>/`
    - `01_input.png`
    - `02a_*` to `02d_*`: individual Paddle overlays (3 layout detectors + VL1.5).
    - `02e_paddle4_board.png`: Paddle-only comparison board.
    - `03_dell_layout.png`
    - `04_mineru_layout.png`
    - `05_fused_layout.png` (recommended variant)
    - `05a_fused_no_miner.png` (P2 paddle4+dell)
    - `05b_fused_with_miner.png` (P4 paddle4+dell+mineru)
    - `06_board.png`: multi-model board
    - `07_with_vs_without_miner.png`: side-by-side miner contribution board
    - `README.txt`
  - `main_png/`: convenient fused overlays by slug.
  - `top20_informative/`
    - `ranking_top20.tsv`
    - `pages/` (copied per-page folders for top 20)
    - `README.txt` (scoring formula)

## Fusion Variants

- `S1_paddle_best_single`
- `S2_dell_only`
- `S3_mineru_only`
- `P1_paddle_union4`
- `P2_paddle_union4_plus_dell`
- `P3_paddle_union4_plus_mineru`
- `P4_paddle_union4_plus_dell_plus_mineru`

## Pipeline Stage Names

- `paddle_layout`
- `paddle_vl15`
- `dell`
- `mineru`
- `fusion`
- `review`
- `transcription`

## Normalized Box Schema

Each normalized box follows:

```json
{
  "source_family": "paddle|dell|mineru",
  "source_model": "variant_id",
  "source_label": "raw model label",
  "norm_label": "text|title|table|image|other",
  "bbox_xyxy": [x1, y1, x2, y2],
  "score": 0.0,
  "reading_order": null,
  "text": null
}
```
