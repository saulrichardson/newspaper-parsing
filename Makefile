VAST_ROOT ?= /vast/sxr203/newspaper-downloads/dedupe-webp
CHUNKS ?= 200
LAYOUT_MANIFEST ?= $(VAST_ROOT)/layouts_all.txt
LAYOUT_PREFIX ?= $(VAST_ROOT)/layouts_chunk_

.PHONY: manifest shard submit

manifest:
	bash scripts/make_layout_manifest.sh $(VAST_ROOT) $(LAYOUT_MANIFEST)

shard: manifest
	bash scripts/split_manifest.sh $(LAYOUT_MANIFEST) $(CHUNKS) $(LAYOUT_PREFIX)

submit: shard
	LAYOUTS_PREFIX=$(LAYOUT_PREFIX) sbatch slurm/vlm_gemini25_array.sbatch
