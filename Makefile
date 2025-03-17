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
RESULT2_BASENAME := "result2"
RESULT_FILE_EXT := ".txt"

FIGS_BASENAME := "result"
FIG_FILE_EXT := ".png"

WEIGHTS_DIR := $(DATA_DIR)/weights
WEIGHTS_BASENAME := "weights"
WEIGHTS_FILE_EXT := ".pt"

MASKS_DIR := $(DATA_DIR)/masks
MASKS_BASENAME := "mask"
MASKS_FILE_EXT := ".pt"

RECONSTRUCTED_DIR := $(DATA_DIR)/reconstructed
RECONSTRUCTED_BASENAME := "reconstructed"
RECONSTRUCTED_FILE_EXT := ".pt"

CUTTED_IMAGES_DIR := $(DATA_DIR)/cutted_images
CUTTED_IMAGES_BASENAME := "cutted_images"
CUTTED_TXT_NAME := "explanatory"
CUTTED_IMAGES_FILE_EXT := ".pt"

IDX_CUTTED = $(shell i=0; while [ -e "$(CUTTED_IMAGES_DIR)/$(CUTTED_IMAGES_BASENAME)_$$i$(CUTTED_IMAGES_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
IDX_CUTTED_MINUS_ONE = $(shell echo $$(($(IDX_CUTTED) - 1)))
CURRENT_CUTTED_IMAGES_PATH = $(CUTTED_IMAGES_DIR)/$(CUTTED_IMAGES_BASENAME)_$(IDX_CUTTED_MINUS_ONE)$(CUTTED_IMAGES_FILE_EXT)
NEXT_CUTTED_IMAGES_PATH = $(CUTTED_IMAGES_DIR)/$(CUTTED_IMAGES_BASENAME)_$(IDX_CUTTED)$(CUTTED_IMAGES_FILE_EXT)
CUTTED_TXT_PATH = $(CUTTED_IMAGES_DIR)/$(CUTTED_TXT_NAME)_$(IDX_CUTTED).txt

# Find the next RESULT_FILE_EXT available filename
IDX=$(shell i=0; while [ -e "$(RESULTS_DIR)/$(RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
RESULT_PATH = $(RESULTS_DIR)/$(RESULT_BASENAME)_$(IDX)$(RESULT_FILE_EXT)
FIGS_PATH = $(FIG_DIR)/$(FIGS_BASENAME)_$(IDX)$(FIG_FILE_EXT)
WEIGHTS_PATH = $(WEIGHTS_DIR)/$(WEIGHTS_BASENAME)_$(IDX)$(WEIGHTS_FILE_EXT)

IDX2 = $(shell i=0; while [ -e "$(RESULTS_DIR)/$(RESULT2_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
RESULT2_PATH = $(RESULTS_DIR)/$(RESULT2_BASENAME)_$(IDX2)$(RESULT_FILE_EXT)

PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json


.PHONY: config preprocess cut1 cut2 train bottleneck test help train2

config:
	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"data\": {" >> $(PATHS_FILE)
	@echo "        \"raw_data_dir\": \"$(DATA_DIR)/raw\"," >> $(PATHS_FILE)
	@echo "        \"processed_data_dir\": \"$(DATA_DIR)/processed/\"," >> $(PATHS_FILE)
	@echo "        \"processed_data_ext\": \"$(PROCESSED_DATA_EXT)\"," >> $(PATHS_FILE)
	@echo "        \"masks_dir\": \"$(MASKS_DIR)\"," >> $(PATHS_FILE)
	@echo "        \"masks_basename\": \"$(MASKS_BASENAME)\"," >> $(PATHS_FILE)
	@echo "        \"masks_file_ext\": \"$(MASKS_FILE_EXT)\"," >> $(PATHS_FILE)
	@echo "        \"current_cutted_images_path\": \"$(CURRENT_CUTTED_IMAGES_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"next_cutted_images_path\": \"$(NEXT_CUTTED_IMAGES_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"cutted_txt_path\": \"$(CUTTED_TXT_PATH)\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"results\": {" >> $(PATHS_FILE)
	@echo "        \"results_path\": \"$(RESULT_PATH)\", " >> $(PATHS_FILE)
	@echo "        \"results2_path\": \"$(RESULT2_PATH)\", " >> $(PATHS_FILE)
	@echo "        \"figs_path\": \"$(FIGS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"weights_path\": \"$(WEIGHTS_PATH)\"" >> $(PATHS_FILE)   
	@echo "    }" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

preprocess: config
	@echo "Preprocessing data..."
	@$(PYTHON) $(PREPROCESSING_DIR)/netcdf_to_torch.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

cut1: config
	@echo "Cutting images..."
	@$(PYTHON) $(PREPROCESSING_DIR)/cut_images_v1.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

cut2: config
	@echo "Cutting images..."
	@$(PYTHON) $(PREPROCESSING_DIR)/cut_images_v2.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

train: config
	@$(PYTHON) $(SRC_DIR)/train_v1.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

train2: config
	@$(PYTHON) $(SRC_DIR)/train_v2.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

bottleneck2: config
	@$(PYTHON) -m torch.utils.bottleneck $(SRC_DIR)/train_v2.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

bottleneck: config
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