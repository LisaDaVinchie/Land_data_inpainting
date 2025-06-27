BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
FIG_DIR := $(BASE_DIR)/figs
TEST_DIR := $(BASE_DIR)/tests

RESULTS_DIR := $(DATA_DIR)/results
RESULTS_DIR_HPC := $(DATA_DIR)/results_dem
RESULT_BASENAME := result
RESULT_FILE_EXT := .txt

WEIGHTS_DIR := $(DATA_DIR)/weights
WEIGHTS_DIR_HPC := $(DATA_DIR)/weights_dem
WEIGHTS_BASENAME := weights
WEIGHTS_FILE_EXT := .pt

RECONSTRUCTED_DIR := $(DATA_DIR)/reconstructed
RECONSTRUCTED_BASENAME := reconstructed
RECONSTRUCTED_FILE_EXT := .pt

DATASETS_DIR := $(DATA_DIR)/minimal_datasets
DATASET_BASENAME := dataset
DATASET_FILE_EXT := .pt

DATASET_SPECS_DIR := $(DATA_DIR)/datasets_specs
DATASET_SPECS_BASENAME := "dataset_specs"
DATASET_SPECS_FILE_EXT := ".json"

OPTIM_DIR = $(DATA_DIR)/optim
OPTIM_DIR_HPC = $(DATA_DIR)/optim_dem
OPTIM_BASENAME = optim
OPTIM_FILE_EXT = .txt
STUDY_BASENAME = study
STUDY_FILE_EXT = .db

CUTTED_IMAGES_DIR := $(DATA_DIR)/cutted_images
CUTTED_IMAGES_BASENAME := cutted_images
CUTTED_TXT_NAME := explanatory
CUTTED_IMAGES_FILE_EXT := .pt

FIG_RESULTS_DIR := $(FIG_DIR)/results
FIG_RESULTS_DIR_HPC := $(FIG_DIR)/results_dem
FIGS_BASENAME := "result"
FIG_FILE_EXT := ".png"

MINMAX_DIR := $(DATA_DIR)/minmax_vals
MINMAX_BASENAME := "minmax"
MINMAX_FILE_EXT := ".pt"

# Find the next available dataset index
CURRENT_IDX_DATASET := $(shell find "$(DATASETS_DIR)" -type f -name "$(DATASET_BASENAME)_*$(DATASET_FILE_EXT)" | \
    sed 's|.*_\([0-9]*\)\$(DATASET_FILE_EXT)|\1|' | \
    sort -n | tail -1)
NEXT_IDX_DATASET = $(shell echo $$(($(CURRENT_IDX_DATASET) + 1)))

CURRENT_MINIMAL_DATASET_PATH = $(DATASETS_DIR)/$(DATASET_BASENAME)_$(CURRENT_IDX_DATASET)$(DATASET_FILE_EXT)
NEXT_MINIMAL_DATASET_PATH = $(DATASETS_DIR)/$(DATASET_BASENAME)_$(NEXT_IDX_DATASET)$(DATASET_FILE_EXT)

CURRENT_DATASET_SPECS_PATH = $(DATASET_SPECS_DIR)/$(DATASET_SPECS_BASENAME)_$(CURRENT_IDX_DATASET)$(DATASET_SPECS_FILE_EXT)
NEXT_DATASET_SPECS_PATH = $(DATASET_SPECS_DIR)/$(DATASET_SPECS_BASENAME)_$(NEXT_IDX_DATASET)$(DATASET_SPECS_FILE_EXT)

CURRENT_NANS_MASKS_PATH = $(NANS_MASKS_DIR)/$(NANS_MASKS_BASENAME)_$(CURRENT_IDX_DATASET)$(NANS_MASKS_FILE_EXT)
NEXT_NANS_MASKS_PATH = $(NANS_MASKS_DIR)/$(NANS_MASKS_BASENAME)_$(NEXT_IDX_DATASET)$(NANS_MASKS_FILE_EXT)

# Find the next available filename
CURRENT_RESULT_IDX := $(shell find "$(RESULTS_DIR)" -type f -name "$(RESULT_BASENAME)_*$(RESULT_FILE_EXT)" | \
    sed 's|.*_\([0-9]*\)\$(RESULT_FILE_EXT)|\1|' | \
    sort -n | tail -1)
NEXT_RESULT_IDX = $(shell echo $$(($(CURRENT_RESULT_IDX) + 1)))
NEXT_RESULT_PATH = $(RESULTS_DIR)/$(RESULT_BASENAME)_$(NEXT_RESULT_IDX)$(RESULT_FILE_EXT)
CURRENT_RESULT_PATH = $(RESULTS_DIR)/$(RESULT_BASENAME)_$(CURRENT_RESULT_IDX)$(RESULT_FILE_EXT)
FIGS_PATH = $(FIG_RESULTS_DIR)/$(FIGS_BASENAME)_$(CURRENT_RESULT_IDX)$(FIG_FILE_EXT)
WEIGHTS_PATH = $(WEIGHTS_DIR)/$(WEIGHTS_BASENAME)_$(NEXT_RESULT_IDX)$(WEIGHTS_FILE_EXT)

CURRENT_RESULT_HPC_IDX := $(shell find "$(RESULTS_DIR_HPC)" -type f -name "$(RESULT_BASENAME)_*$(RESULT_FILE_EXT)" | \
    sed 's|.*_\([0-9]*\)\$(RESULT_FILE_EXT)|\1|' | \
    sort -n | tail -1)

CURRENT_RESULT_HPC_PATH = $(RESULTS_DIR_HPC)/$(RESULT_BASENAME)_$(CURRENT_RESULT_HPC_IDX)$(RESULT_FILE_EXT)

# Find the next available optimization index
CURRENT_STUDY_IDX := $(shell find "$(OPTIM_DIR)" -type f -name "$(STUDY_BASENAME)_*$(STUDY_FILE_EXT)" | \
    sed 's|.*_\([0-9]*\)\$(STUDY_FILE_EXT)|\1|' | \
    sort -n | tail -1)
NEXT_STUDY_IDX = $(shell echo $$(($(CURRENT_STUDY_IDX) + 1)))
# IDX=$(shell i=0; while [ -e "$(OPTIM_DIR)/$(STUDY_BASENAME)_$$i$(STUDY_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
NEXT_OPTIM_PATH = $(OPTIM_DIR)/$(OPTIM_BASENAME)_$(NEXT_STUDY_IDX)$(OPTIM_FILE_EXT)
NEXT_STUDY_PATH = $(OPTIM_DIR)/$(STUDY_BASENAME)_$(NEXT_STUDY_IDX)$(STUDY_FILE_EXT)
CURRENT_OPTIM_PATH = $(OPTIM_DIR)/$(OPTIM_BASENAME)_$(CURRENT_STUDY_IDX)$(OPTIM_FILE_EXT)
CURRENT_STUDY_PATH = $(OPTIM_DIR)/$(STUDY_BASENAME)_$(CURRENT_STUDY_IDX)$(STUDY_FILE_EXT)


PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json
OPTIM_PARAMS_FILE := $(SRC_DIR)/optim_params.json


.PHONY: config train btrain optim loss test plot plothpc clean help

config:
	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"data\": {" >> $(PATHS_FILE)
	@echo "        \"current_minimal_dataset_path\": \"$(CURRENT_MINIMAL_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"next_minimal_dataset_path\": \"$(NEXT_MINIMAL_DATASET_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"current_dataset_specs_path\": \"$(CURRENT_DATASET_SPECS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"dataset_specs_path\": \"$(NEXT_DATASET_SPECS_PATH)\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"results\": {" >> $(PATHS_FILE)
	@echo "        \"results_path\": \"$(NEXT_RESULT_PATH)\", " >> $(PATHS_FILE)
	@echo "        \"current_results_path\": \"$(CURRENT_RESULT_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"figs_path\": \"$(FIGS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"weights_path\": \"$(WEIGHTS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"current_optim_path\": \"$(CURRENT_OPTIM_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"current_study_path\": \"$(CURRENT_STUDY_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"optim_next_path\": \"$(NEXT_OPTIM_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"study_next_path\": \"$(NEXT_STUDY_PATH)\"" >> $(PATHS_FILE)
	@echo "    }" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

train: config
	@$(PYTHON) $(SRC_DIR)/train.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

train1: config
	@$(PYTHON) $(SRC_DIR)/train1.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

train2: config
	@$(PYTHON) $(SRC_DIR)/train2.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

btrain: config
	@$(PYTHON) -m torch.utils.bottleneck $(SRC_DIR)/train4.py --params $(PARAMS_FILE) --paths $(PATHS_FILE)

plot:config
	@echo "Plotting results..."
	@$(PYTHON) $(SRC_DIR)/plot_results.py --respath $(CURRENT_RESULT_PATH) --figdir $(FIG_RESULTS_DIR)

loss: config
	@echo "Plotting loss..."
	@$(PYTHON) $(SRC_DIR)/calculate_dataset_mean_loss.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)

plothpc: config
	@echo "Plotting results for HPC..."
	@$(PYTHON) $(SRC_DIR)/plot_results.py --respath $(CURRENT_RESULT_HPC_PATH) --figdir $(FIG_RESULTS_DIR_HPC)

optim: config
	@echo "Optimizing model with new parameters..."
	@echo "current study idx is $(CURRENT_STUDY_IDX)"
	@$(PYTHON) $(SRC_DIR)/params_optimization.py --params $(OPTIM_PARAMS_FILE) --paths $(PATHS_FILE)

test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*_test.py"

clean: config
	@echo "Removing dataset $(CURRENT_MINIMAL_DATASET_PATH)"
	rm -rf $(CURRENT_MINIMAL_DATASET_PATH) $(CURRENT_NANS_MASKS_PATH) $(CURRENT_DATASET_SPECS_PATH)

help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  config: Store paths to json"
	@echo "  train: Train the model"
	@echo "  btrain: Run the bottleneck profiler on the training script"
	@echo "  test: Run tests"
	@echo "  plot: Plot results"
	@echo "  clean: Remove the last dataset and its related files"
	@echo "  help: Display this help message"