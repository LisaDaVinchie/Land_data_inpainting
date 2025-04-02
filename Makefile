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

WEIGHTS_DIR := $(DATA_DIR)/weights
WEIGHTS_BASENAME := "weights"
WEIGHTS_FILE_EXT := ".pt"

MASKS_DIR := $(DATA_DIR)/masks
MASKS_BASENAME := "mask"
MASKS_FILE_EXT := ".pt"

RECONSTRUCTED_DIR := $(DATA_DIR)/reconstructed
RECONSTRUCTED_BASENAME := "reconstructed"
RECONSTRUCTED_FILE_EXT := ".pt"

MINIMAL_DATASETS_DIR := $(DATA_DIR)/minimal_datasets
EXTENDED_DATASETS_DIR := $(DATA_DIR)/extended_datasets
DATASET_BASENAME := "dataset"
DATASET_FILE_EXT := ".pt"

DATASET_SPECS_DIR := $(DATA_DIR)/datasets_specs
DATASET_SPECS_BASENAME := "dataset_specs"
DATASET_SPECS_FILE_EXT := ".txt"

CUTTED_IMAGES_DIR := $(DATA_DIR)/cutted_images
CUTTED_IMAGES_BASENAME := "cutted_images"
CUTTED_TXT_NAME := "explanatory"
CUTTED_IMAGES_FILE_EXT := ".pt"

NANS_MASKS_DIR := $(DATA_DIR)/nans_masks
NANS_MASKS_BASENAME := "nans_mask"
NANS_MASKS_FILE_EXT := ".pt"

FIG_RESULTS_DIR := $(FIG_DIR)/results
FIGS_BASENAME := "result"
FIG_FILE_EXT := ".png"

MINMAX_DIR := $(DATA_DIR)/minmax_vals
MINMAX_BASENAME := "minmax"
MINMAX_FILE_EXT := ".pt"

# Find the next available dataset index
IDX_DATASET = $(shell i=0; while [ -e "$(MINIMAL_DATASETS_DIR)/$(DATASET_BASENAME)_$$i$(DATASET_FILE_EXT)" ] || [ -e "$(EXTENDED_DATASETS_DIR)/$(DATASET_BASENAME)_$$i$(DATASET_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
IDX_DATASET_MINUS_ONE = $(shell echo $$(($(IDX_DATASET) - 1)))
NEXT_MINIMAL_DATASET_PATH = $(MINIMAL_DATASETS_DIR)/$(DATASET_BASENAME)_$(IDX_DATASET)$(DATASET_FILE_EXT)
NEXT_EXTENDED_DATASET_PATH = $(EXTENDED_DATASETS_DIR)/$(DATASET_BASENAME)_$(IDX_DATASET)$(DATASET_FILE_EXT)
CURRENT_MINIMAL_DATASET_PATH = $(MINIMAL_DATASETS_DIR)/$(DATASET_BASENAME)_$(IDX_DATASET_MINUS_ONE)$(DATASET_FILE_EXT)
CURRENT_EXTENDED_DATASET_PATH = $(EXTENDED_DATASETS_DIR)/$(DATASET_BASENAME)_$(IDX_DATASET_MINUS_ONE)$(DATASET_FILE_EXT)
DATASET_SPECS_PATH = $(DATASET_SPECS_DIR)/$(DATASET_SPECS_BASENAME)_$(IDX_DATASET)$(DATASET_SPECS_FILE_EXT)
CURRENT_NANS_MASKS_PATH = $(NANS_MASKS_DIR)/$(NANS_MASKS_BASENAME)_$(IDX_DATASET_MINUS_ONE)$(NANS_MASKS_FILE_EXT)
NEXT_NANS_MASKS_PATH = $(NANS_MASKS_DIR)/$(NANS_MASKS_BASENAME)_$(IDX_DATASET)$(NANS_MASKS_FILE_EXT)
CURRENT_MIMAX_PATH = $(MINMAX_DIR)/$(MINMAX_BASENAME)_$(IDX_DATASET_MINUS_ONE)$(MINMAX_FILE_EXT)
NEXT_MINMAX_PATH = $(MINMAX_DIR)/$(MINMAX_BASENAME)_$(IDX_DATASET)$(MINMAX_FILE_EXT)

# Find the next RESULT_FILE_EXT available filename
IDX=$(shell i=0; while [ -e "$(RESULTS_DIR)/$(RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
IDX_MINUS_ONE = $(shell echo $$(($(IDX) - 1)))
RESULT_PATH = $(RESULTS_DIR)/$(RESULT_BASENAME)_$(IDX)$(RESULT_FILE_EXT)
CURRENT_RESULT_PATH = $(RESULTS_DIR)/$(RESULT_BASENAME)_$(IDX_MINUS_ONE)$(RESULT_FILE_EXT)
FIGS_PATH = $(FIG_RESULTS_DIR)/$(FIGS_BASENAME)_$(IDX)$(FIG_FILE_EXT)
WEIGHTS_PATH = $(WEIGHTS_DIR)/$(WEIGHTS_BASENAME)_$(IDX)$(WEIGHTS_FILE_EXT)

PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json


.PHONY: config preprocess cut train bottleneck test plot help

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
	@echo "        \"next_extended_dataset_path\": \"$(NEXT_EXTENDED_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"current_extended_dataset_path\": \"$(CURRENT_EXTENDED_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"current_minimal_dataset_path\": \"$(CURRENT_MINIMAL_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"next_minimal_dataset_path\": \"$(NEXT_MINIMAL_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"dataset_specs_path\": \"$(DATASET_SPECS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"current_nans_masks_path\": \"$(CURRENT_NANS_MASKS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"next_nans_masks_path\": \"$(NEXT_NANS_MASKS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"current_minmax_path\": \"$(CURRENT_MIMAX_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"next_minmax_path\": \"$(NEXT_MINMAX_PATH)\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"results\": {" >> $(PATHS_FILE)
	@echo "        \"results_path\": \"$(RESULT_PATH)\", " >> $(PATHS_FILE)
	@echo "        \"current_results_path\": \"$(CURRENT_RESULT_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"figs_path\": \"$(FIGS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"weights_path\": \"$(WEIGHTS_PATH)\"" >> $(PATHS_FILE)   
	@echo "    }" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

preprocess: config
	@echo "Preprocessing data..."
	@$(PYTHON) $(PREPROCESSING_DIR)/netcdf_to_torch.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

cut: config
	@echo "Cutting images..."
	@$(PYTHON) $(PREPROCESSING_DIR)/cut_images.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

train: config
	@$(PYTHON) $(SRC_DIR)/train.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

bottleneck: config
	@$(PYTHON) -m torch.utils.bottleneck $(SRC_DIR)/train.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

plot:config
	@echo "Plotting results..."
	@$(PYTHON) $(SRC_DIR)/plot_results.py --paths $(PATHS_FILE)

test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*_test.py"

help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  config: Store paths to json"
	@echo "  preprocess: Preprocess data"
	@echo "  cut: create a dataset of images"
	@echo "  mask: Create masks"
	@echo "  train: Train the model"
	@echo "  test: Run tests"
	@echo "  bottleneck: Run the bottleneck profiler on the training script"
	@echo "  help: Display this help message"