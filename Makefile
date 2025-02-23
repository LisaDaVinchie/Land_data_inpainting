BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

DATA_DIR := $(BASE_DIR)/data
SRC_DIR := $(BASE_DIR)/src
FIG_DIR := $(BASE_DIR)/figs
TEST_DIR := $(BASE_DIR)/tests

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