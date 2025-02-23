BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
FIG_DIR := $(BASE_DIR)/figs
TEST_DIR := $(BASE_DIR)/tests

RESULTS_DIR := $(DATA_DIR)/results
RESULT_BASENAME := "result"
RESULT_FILE_EXT := ".txt"

FIGS_BASENAME := "result"
FIG_FILE_EXT := ".png"

WEIGHTS_DIR := $(DATA_DIR)/weights
WEIGHTS_BASENAME := "weights"
WEIGHTS_FILE_EXT := ".pth"

# Find the nRESULT_FILE_EXT available filename
IDX=$(shell i=0; while [ -e "$(RESULTS_DIR)/$(RESULT_BASENAME)_$$i$(RESULT_FILE_EXT)" ]; do i=$$((i+1)); done; echo "$$i")
RESULT_PATH = $(RESULTS_DIR)/$(RESULT_BASENAME)_$(IDX)$(RESULT_FILE_EXT)
FIGS_PATH = $(FIG_DIR)/$(FIGS_BASENAME)_$(IDX)$(FIG_FILE_EXT)
WEIGHTS_PATH = $(WEIGHTS_DIR)/$(WEIGHTS_BASENAME)_$(IDX)$(WEIGHTS_FILE_EXT)


PATHS_FILE := $(SRC_DIR)/paths.json
PARAMS_FILE := $(SRC_DIR)/params.json

DATASET_NAME := "IFREMER-GLOB-SST-L3-NRT-OBS_FULL_TIME_SERIE_1740148697395"


.PHONY: config preprocess train test help

config:
	@echo "Storing paths to json..."
	@echo "{" > $(PATHS_FILE)
	@echo "    \"data\": {" >> $(PATHS_FILE)
	@echo "        \"raw_data_path\": \"$(DATA_DIR)/raw/$(DATASET_NAME).nc\"," >> $(PATHS_FILE)
	@echo "        \"processed_data_path\": \"$(DATA_DIR)/processed/$(DATASET_NAME).pth\"" >> $(PATHS_FILE)
	@echo "    }," >> $(PATHS_FILE)
	@echo "    \"results\": {" >> $(PATHS_FILE)
	@echo "        \"results_path\": \"$(RESULT_PATH)\", " >> $(PATHS_FILE)
	@echo "        \"figs_path\": \"$(FIGS_PATH)\"," >> $(PATHS_FILE)
	@echo "        \"weights_path\": \"$(WEIGHTS_PATH)\"" >> $(PATHS_FILE)   
	@echo "    }" >> $(PATHS_FILE)
	@echo "}" >> $(PATHS_FILE)

preprocess: config
	@echo "Preprocessing data..."
	@$(PYTHON) $(SRC_DIR)/netcdf_to_torch.py --paths $(PATHS_FILE)

train: config
	@$(PYTHON) $(SRC_DIR)/train.py --paths $(PATHS_FILE) --params $(PARAMS_FILE)
test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s $(TEST_DIR) -p "*_test.py"