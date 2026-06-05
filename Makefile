# CaloEmbed — common development tasks
#
# Nsight profiling targets require nsys / ncu to be on PATH.
# On lxplus/CERN clusters: source /cvmfs/sft.cern.ch/lcg/... or module load cuda

DATA    ?= /vols/cms/mm1221/cms/Data/100k/hdf5
CONFIG  ?= configs/raw.yaml
OUTPUT  ?= results/raw
BACKEND ?= auto

.PHONY: install test run profile profile-kernels clean

ROOT_DIR      ?= /vols/cms/mm1221/cms/Data/100k/root
HDF5_DIR      ?= /vols/cms/mm1221/cms/Data/100k/hdf5
JOBS          ?= 32
SYSTEM_PYTHON ?= /usr/bin/python3   # must have ROOT; not the micromamba env

install:
	pip install -e .
	pip install -e CLUEstering/

test:
	PYTHONPATH=$(PWD) python3 -m pytest tests/ -v -m "not integration"

test-all:
	PYTHONPATH=$(PWD) python3 -m pytest tests/ -v

preprocess:
	PYTHONPATH=$(PWD) $(SYSTEM_PYTHON) -m caloembed.scripts.preprocess_root \
		--input-dir $(ROOT_DIR) \
		--output-dir $(HDF5_DIR) \
		--jobs $(JOBS)

preprocess-one:
	PYTHONPATH=$(PWD) $(SYSTEM_PYTHON) -m caloembed.scripts.preprocess_root \
		--input $(ROOT_DIR)/histo_921.root \
		--output $(HDF5_DIR)/histo_921.h5 \
		--verbose

preprocess-check:
	@echo "ROOT files : $$(ls $(ROOT_DIR)/*.root 2>/dev/null | wc -l)"
	@echo "HDF5 done  : $$(ls $(HDF5_DIR)/*.h5   2>/dev/null | wc -l)"
	@echo "Remaining  : $$(comm -23 \
		<(ls $(ROOT_DIR)/*.root | xargs -n1 basename | sed 's/.root/.h5/' | sort) \
		<(ls $(HDF5_DIR)/*.h5   | xargs -n1 basename | sort) \
		| wc -l)"

run:
	caloembed-run \
		--config $(CONFIG) \
		--data   $(DATA) \
		--output $(OUTPUT) \
		--backend "$(BACKEND)"

# Nsight Systems: full timeline (CPU + GPU + NVTX ranges)
# Output: $(OUTPUT)/nsys_profile.nsys-rep  →  open with nsys-ui
profile:
	nsys profile \
		--trace=cuda,nvtx,osrt \
		--output=$(OUTPUT)/nsys_profile \
		--force-overwrite=true \
		caloembed-run \
			--config $(CONFIG) \
			--data   $(DATA) \
			--output $(OUTPUT)

# Nsight Compute: kernel-level roofline + memory analysis
# Output: $(OUTPUT)/ncu_profile.ncu-rep  →  open with ncu-ui
profile-kernels:
	ncu \
		--set full \
		--output $(OUTPUT)/ncu_profile \
		--force-overwrite \
		caloembed-run \
			--config $(CONFIG) \
			--data   $(DATA) \
			--output $(OUTPUT)

clean:
	find results/ -mindepth 2 -not -name '.gitkeep' -delete
