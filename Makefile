BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATASET_NAME := "IFREMER-GLOB-SST-L3-NRT-OBS_FULL_TIME_SERIE_202211"

DATA_DIR := $(BASE_DIR)/data/$(DATASET_NAME)
SRC_DIR := $(BASE_DIR)/src
FIG_DIR := $(BASE_DIR)/figs
TEST_DIR := $(BASE_DIR)/tests

PREPROCESSING_DIR := $(SRC_DIR)/preprocessing

PROCESSED_DATA_EXT := ".pt"

RESULTS_DIR := $(DATA_DIR)/results
RESULT_BASENAME := "result"
RESULT_FILE_EXT := ".txt"

FIGS_BASENAME := "result"
FIG_FILE_EXT := ".png"

WEIGHTS_DIR := $(DATA_DIR)/weights
WEIGHTS_BASENAME := "weights"
WEIGHTS_FILE_EXT := ".pt"

MASKS_DIR := $(DATA_DIR)/masks
MASKS_BASENAME := "mask"
MASKS_FILE_EXT := ".pt"

# Find the nRESULT_FILE_EXT available filename
IDX=$(shell i=0; while [ -e "$(RESULTS_DIR)/$(RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
RESULT_PATH = $(RESULTS_DIR)/$(RESULT_BASENAME)_$(IDX)$(RESULT_FILE_EXT)
FIGS_PATH = $(FIG_DIR)/$(FIGS_BASENAME)_$(IDX)$(FIG_FILE_EXT)
WEIGHTS_PATH = $(WEIGHTS_DIR)/$(WEIGHTS_BASENAME)_$(IDX)$(WEIGHTS_FILE_EXT)


PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json


.PHONY: config preprocess mask train bottleneck test help

config:
	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"data\": {" >> $(PATHS_FILE)
	@echo "        \"raw_data_dir\": \"$(DATA_DIR)/raw\"," >> $(PATHS_FILE)
	@echo "        \"processed_data_dir\": \"$(DATA_DIR)/processed/\"," >> $(PATHS_FILE)
	@echo "		   \"processed_data_ext\": \"$(PROCESSED_DATA_EXT)\"," >> $(PATHS_FILE)
	@echo "        \"masks_dir\": \"$(MASKS_DIR)\"," >> $(PATHS_FILE)
	@echo "        \"masks_basename\": \"$(MASKS_BASENAME)\"," >> $(PATHS_FILE)
	@echo "        \"masks_file_ext\": \"$(MASKS_FILE_EXT)\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"results\": {" >> $(PATHS_FILE)
	@echo "        \"results_path\": \"$(RESULT_PATH)\", " >> $(PATHS_FILE)
	@echo "        \"figs_path\": \"$(FIGS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"weights_path\": \"$(WEIGHTS_PATH)\"" >> $(PATHS_FILE)   
	@echo "    }" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

preprocess: config
	@echo "Preprocessing data..."
	@$(PYTHON) $(PREPROCESSING_DIR)/netcdf_to_torch.py --paths $(PATHS_FILE)

mask: config
	@echo "Creating masks..."
	@$(PYTHON) $(PREPROCESSING_DIR)/create_masks.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

train: config
	@$(PYTHON) $(SRC_DIR)/train.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

train: bottleneck
	@$(PYTHON) -m torch.utils.bottleneck $(SRC_DIR)/train.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*_test.py"

help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  config: Store paths to json"
	@echo "  preprocess: Preprocess data"
	@echo "  mask: Create masks"
	@echo "  train: Train the model"
	@echo "  test: Run tests"
	@echo "  bottleneck: Run the bottleneck profiler on the training script"
	@echo "  help: Display this help message"