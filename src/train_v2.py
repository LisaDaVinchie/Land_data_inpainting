import torch as th
from pathlib import Path
import argparse
from models import simple_conv, conv_maxpool, conv_unet, DINCAE_like
from time import time
import json
import psutil
import os

from preprocessing.mask_data import mask_inversemask_image
from utils.create_dataloaders_v2 import create_dataloaders
from utils.import_params_json import load_config

start_time = time()


def track_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1e6  # Convert to MB
    print(f"[Memory] {stage}: {mem_info:.2f} MB\n", flush=True)

parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')
parser.add_argument('--dataset_idx', type=int, help='Index of the dataset to use', required=False)

args = parser.parse_args()

params_path = args.params
paths_path = args.paths

# Load the paths
results2_path: Path = None
figs_path: Path = None
weights_path: Path = None
processed_data_dir: Path = None
masks_dir: Path = None
current_cutted_images_path: Path = None
cutted_images_basename: str = None
cutted_images_file_ext: str = None

n_channels: int = None
paths = load_config(paths_path, ["data", "results"])
locals().update(paths["results"])
locals().update(paths["data"])

results2_path = Path(results2_path)
figs_path = Path(figs_path)
weights_path = Path(weights_path)
processed_data_dir = Path(processed_data_dir)
masks_dir = Path(masks_dir)
current_cutted_images_path = Path(current_cutted_images_path)

for path in [results2_path.parent, figs_path.parent, weights_path.parent, current_cutted_images_path]:
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

print("\nPaths imported\n", flush = True)


# Load the train parameters
train_perc: int = None
epochs: int = None
batch_size: int = None
learning_rate: float = None
optimizer_kind: str = None
loss_kind: str = None
model_kind: str = None
placeholder: float = None
cutted_width: int = None
cutted_height: int = None

n_channels: int = None
n_cutted_images: int = None
config = load_config(params_path, ["training", "dataset"])
locals().update(config["training"])
locals().update(config["dataset"])
print("Parameters imported\n", flush = True)

dataset_idx = None
if args.dataset_idx is not None:
    dataset_idx = args.dataset_idx
    current_cutted_images_path = current_cutted_images_path.parent / f"cutted_images_{dataset_idx}.pt"

track_memory("Before loading model")
if model_kind == "simple_conv":
    model = simple_conv(params_path)
elif model_kind == "conv_maxpool":
    model = conv_maxpool(params_path)
elif model_kind == "conv_unet":
    model = conv_unet(params_path)
elif model_kind == "DINCAE_like":
    model = DINCAE_like(params_path, image_width=cutted_width, image_height=cutted_height, n_channels=n_channels)
else:
    raise ValueError(f"Model kind {model_kind} not recognized")
track_memory("After loading model")

print("Model initialized\n", flush = True)

dataset_start_time = time()
track_memory("Before loading dataset")
dataset = th.load(current_cutted_images_path)
track_memory("After loading dataset")

print(f"Dataset created in {time() - dataset_start_time:.2f} seconds", flush = True)

track_memory("Before creating dataloaders")
train_loader, test_loader = create_dataloaders(masked_images=dataset["masked_images"], inverse_masked_images=dataset["inverse_masked_images"], masks=dataset["masks"], train_perc=train_perc, batch_size=batch_size)
del dataset
print("Dataloaders created\n", flush = True)
track_memory("After creating dataloaders")

loss_function = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []

track_memory("Before training")
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}\n", flush=True)
    model.train()
    
    train_loss = 0
    for i, (masked_image, inverse_masked_image, mask) in enumerate(train_loader):
        output = model(masked_image)
        _, inverse_masked_output = mask_inversemask_image(output, mask, placeholder)
        loss_val = loss_function(inverse_masked_output, inverse_masked_image) / th.sum(mask)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        train_loss += loss_val.item()
    
    train_losses.append(train_loss / len(train_loader))
    
    with th.no_grad():
        model.eval()
        test_loss = 0
        for i, (masked_image, inverse_masked_image, mask) in enumerate(test_loader):
            output = model(masked_image)
            _, inverse_masked_output = mask_inversemask_image(output, mask, placeholder)
            loss_val = loss_function(inverse_masked_output, inverse_masked_image) / th.sum(mask)
            test_loss += loss_val.item()
        
        test_losses.append(test_loss / len(test_loader))

elapsed_time = time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

track_memory("After training")

# Save the model
th.save(model.state_dict(), weights_path)

with open(params_path, 'r') as f:
    params = json.load(f)
    
json_str = json.dumps(params, indent=4)[1: -1]

# Save the train losses to a txt file
with open(results2_path, 'w') as f:
    f.write("Elapsed time [s]:\n")
    f.write(f"{elapsed_time}\n\n")
    f.write("Used dataset:\n")
    f.write(f"{current_cutted_images_path.relative_to(current_cutted_images_path.parent.parent)}\n\n")
    f.write("Train losses\n")
    for i, loss_function in enumerate(train_losses):
        f.write(f"{loss_function}\t")
    f.write("\n\n")
    f.write("Test losses\n")
    for i, loss_function in enumerate(test_losses):
        f.write(f"{loss_function}\t")
    f.write("\n\n")
    f.write("Parameters\n")
    f.write(json_str)
    
        
