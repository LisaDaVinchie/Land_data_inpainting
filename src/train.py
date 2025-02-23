import torch as th
from pathlib import Path
import argparse
from models import simple_conv, conv_maxpool
from utils.import_params_json import load_config

parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

args = parser.parse_args()

params_path = args.params
paths_path = args.paths

# Load the train parameters

epochs: int = None
batch_size: int = None
learning_rate: float = None
optimizer_kind: str = None
loss_kind: str = None
model_kind: str = None
config = load_config(params_path, ["training"])
locals().update(config["training"])

