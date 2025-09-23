#!/bin/bash

REMOTE_USER="s237347@demetra.units.it"
BASE_PATH="data/minimal_datasets/"

REMOTE_BASE_PATH="/u/ldavinchie/Land_data_inpainting/${BASE_PATH}"

LOCAL_TEST_DS="$BASE_PATH/dataset_proc_1_test.nc"
LOCAL_TRAIN_DS="$BASE_PATH/dataset_proc_1.nc"

# Run the rsync command
rsync -r ${LOCAL_TEST_DS} ${REMOTE_USER}:${REMOTE_BASE_PATH}
rsync -r -z ${LOCAL_TRAIN_DS} ${REMOTE_USER}:${REMOTE_BASE_PATH}
