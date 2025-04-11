#!/bin/bash

# Create the folders 'data' and 'figs'
mkdir -p data figs

# Create subfolders inside 'data'
mkdir -p data/dataset_specs data/minimal_datasets data/minmax_vals data/nans_masks \
         data/processed data/raw data/reconstructed data/results data/weights

# Create subfolders inside 'figs'
mkdir -p figs/reconstructed figs/results