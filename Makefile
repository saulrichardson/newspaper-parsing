VAST_ROOT ?= /vast/sxr203/newspaper-downloads/dedupe-webp
CHUNKS ?= 200
LAYOUT_MANIFEST ?= $(VAST_ROOT)/layouts_all.txt
LAYOUT_PREFIX ?= $(VAST_ROOT)/layouts_chunk_

.PHONY: manifest shard submit
.PHONY: questionnaire-excel
.PHONY: questionnaire-pca
.PHONY: questionnaire-timeseries

manifest:
	bash scripts/make_layout_manifest.sh $(VAST_ROOT) $(LAYOUT_MANIFEST)

shard: manifest
	bash scripts/split_manifest.sh $(LAYOUT_MANIFEST) $(CHUNKS) $(LAYOUT_PREFIX)

submit: shard
	LAYOUTS_PREFIX=$(LAYOUT_PREFIX) sbatch slurm/vlm_gemini25_array.sbatch

# Build a merged Excel workbook with ordinance questionnaire answers + stats.
#
# Usage:
#   make questionnaire-excel OUT_XLSX=path/to/out.xlsx
# Optional:
#   QUESTIONS_XLSX=~/Downloads/Questions.xlsx
OUT_XLSX ?= newspaper-parsing-local/data/questionnaire_answers_latest.xlsx
QUESTIONS_XLSX ?= $(HOME)/Downloads/Questions.xlsx
SKIP_RUN2 ?= 0

# Optional overrides if you move/rename runs.
RUN1_REQ_DIR ?= newspaper-parsing-local/data/batch_requests_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_025258
RUN1_RES_DIR ?= newspaper-parsing-local/data/batch_results_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_025258
RUN2_REQ_DIR ?= newspaper-parsing-local/data/batch_requests_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_062159_incremental
RUN2_RES_DIR ?= newspaper-parsing-local/data/batch_results_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_062159_incremental

questionnaire-excel:
	QUESTIONS_XLSX="$(QUESTIONS_XLSX)" \
	RUN1_REQ_DIR="$(RUN1_REQ_DIR)" RUN1_RES_DIR="$(RUN1_RES_DIR)" \
	RUN2_REQ_DIR="$(RUN2_REQ_DIR)" RUN2_RES_DIR="$(RUN2_RES_DIR)" \
	SKIP_RUN2="$(SKIP_RUN2)" \
	bash scripts/refresh_ordinance_questionnaire_excel.sh "$(OUT_XLSX)"

# Run PCA over the questionnaire outputs (ai-zoning-style imputation + PCA).
#
# Usage:
#   make questionnaire-pca
# Optional:
#   PCA_GROUP_KEY=slug|page_id
#   PCA_OUT_XLSX=path/to/out.xlsx
#   PCA_CATEGORICAL_MODE=drop|onehot
PCA_GROUP_KEY ?= slug
PCA_CATEGORICAL_MODE ?= drop
PCA_OUT_XLSX ?= newspaper-parsing-local/data/questionnaire_pca_$(PCA_GROUP_KEY).xlsx
LOCATIONS_PARQUET ?=
LOCATIONS_JOIN_LEFT ?= slug
LOCATIONS_JOIN_RIGHT ?= pub_slug

ifneq ($(strip $(LOCATIONS_PARQUET)),)
PCA_LOC_ARGS := --locations-parquet "$(LOCATIONS_PARQUET)" --locations-join-left "$(LOCATIONS_JOIN_LEFT)" --locations-join-right "$(LOCATIONS_JOIN_RIGHT)"
else
PCA_LOC_ARGS :=
endif

questionnaire-pca: questionnaire-excel
	python scripts/compute_questionnaire_pca.py \
	  --answers-xlsx "$(OUT_XLSX)" \
	  --questions-xlsx "$(QUESTIONS_XLSX)" \
	  --group-keys "$(PCA_GROUP_KEY)" \
	  $(PCA_LOC_ARGS) \
	  --categorical-mode "$(PCA_CATEGORICAL_MODE)" \
	  --out-xlsx "$(PCA_OUT_XLSX)"

# Rank “interesting” time series from a PCA workbook and optionally render plots.
#
# Usage:
#   make questionnaire-timeseries
# Optional:
#   TS_PCA_XLSX=path/to/questionnaire_pca_city_state_page_year.xlsx
#   TS_OUT_XLSX=path/to/report.xlsx
#   TS_PLOTS_DIR=path/to/plots_dir   (blank disables)
TS_PCA_XLSX ?= newspaper-parsing-local/data/questionnaire_pca_city_state_page_year.xlsx
TS_OUT_XLSX ?= newspaper-parsing-local/data/questionnaire_pca_timeseries_report.xlsx
TS_PLOTS_DIR ?=
TS_OUT_TEX ?=
TS_MIN_YEARS ?= 5
TS_MIN_PAGES_TOTAL ?= 30
TS_MIN_BOXES_TOTAL ?= 0
TS_TOP_N ?= 5
TS_ZONING_REQ_DIR ?=
TS_ZONING_RES_DIR ?=
TS_ZONING_MIN_CONF ?= 0.0

questionnaire-timeseries:
	python scripts/analyze_questionnaire_pca_timeseries.py \
	  --pca-xlsx "$(TS_PCA_XLSX)" \
	  --out-xlsx "$(TS_OUT_XLSX)" \
	  --min-years "$(TS_MIN_YEARS)" \
	  --min-pages-total "$(TS_MIN_PAGES_TOTAL)" \
	  --min-boxes-total "$(TS_MIN_BOXES_TOTAL)" \
	  --top-n "$(TS_TOP_N)" \
	  $(if $(strip $(TS_ZONING_REQ_DIR)),--zoning-classification-request-dir "$(TS_ZONING_REQ_DIR)",) \
	  $(if $(strip $(TS_ZONING_RES_DIR)),--zoning-classification-results-dir "$(TS_ZONING_RES_DIR)",) \
	  $(if $(and $(strip $(TS_ZONING_REQ_DIR)),$(strip $(TS_ZONING_RES_DIR))),--zoning-min-confidence "$(TS_ZONING_MIN_CONF)",) \
	  $(if $(strip $(TS_OUT_TEX)),--out-tex "$(TS_OUT_TEX)",) \
	  $(if $(strip $(TS_PLOTS_DIR)),--plots-dir "$(TS_PLOTS_DIR)",)
