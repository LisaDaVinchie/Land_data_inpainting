import torch as th
from pathlib import Path
import argparse
import json
from time import time

start_time = time()

parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')

args = parser.parse_args()

paths_file_path = Path(args.paths)

with open(paths_file_path, 'r') as f:
    paths = json.load(f)
    
results_path = Path("data/dataset1.pt")
processed_data_dir = Path(paths["data"]["processed_data_dir"])

file_paths = list(processed_data_dir.glob("*.pt"))

dataset = th.stack([th.load(file_path) for file_path in file_paths])

print(dataset.shape)

# th.save(dataset, results_path)
print(f"Time taken: {time() - start_time:.2f}s")