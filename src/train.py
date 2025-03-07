import torch as th
from pathlib import Path
import argparse
from models import simple_conv, conv_maxpool, conv_unet, DINCAE_like
from time import time

from mask_data import apply_mask_on_channel
from utils.create_dataloaders import create_dataloaders
from utils.import_params_json import load_config

start_time = time()

parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

args = parser.parse_args()

params_path = args.params
paths_path = args.paths

# Load the paths
results_path: Path = None
figs_path: Path = None
weights_path: Path = None
processed_data_dir: Path = None
masks_dir: Path = None

n_channels: int = None
paths = load_config(paths_path, ["data", "results"])
locals().update(paths["results"])
locals().update(paths["data"])

results_path = Path(results_path)
figs_path = Path(figs_path)
weights_path = Path(weights_path)
processed_data_dir = Path(processed_data_dir)
masks_dir = Path(masks_dir)

for path in [results_path.parent, figs_path.parent, weights_path.parent, processed_data_dir, masks_dir]:
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

# Load the train parameters
train_perc: int = None
epochs: int = None
batch_size: int = None
learning_rate: float = None
optimizer_kind: str = None
loss_kind: str = None
model_kind: str = None
placeholder: float = None

n_channels: int = None
config = load_config(params_path, ["training", "dataset"])
locals().update(config["training"])
locals().update(config["dataset"])

if model_kind == "simple_conv":
    model = simple_conv(params_path)
elif model_kind == "conv_maxpool":
    model = conv_maxpool(params_path)
elif model_kind == "conv_unet":
    model = conv_unet(params_path)
elif model_kind == "DINCAE_like":
    model = DINCAE_like(params_path)
else:
    raise ValueError(f"Model kind {model_kind} not recognized")

# Import dataset
# dataset = th.rand(100, 13, 10, 10)
# channels_to_mask = [3, 4, 5]
# masks = th.ones(100, 13, 10, 10)

# masks[:, channels_to_mask, :2, :2] = 0

dataset_start_time = time()
data_files = list(processed_data_dir.glob("*.pt"))
mask_files = list(masks_dir.glob("*.pt"))

dataset = th.stack([th.load(file) for file in data_files])
masks = th.load(masks_dir / f"mask_n{len(data_files)}_c{n_channels}.pt")

print(f"Dataset created in {time() - dataset_start_time:.2f} seconds")

train_loader, test_loader = create_dataloaders(dataset, masks, train_perc, batch_size)
del dataset, masks

loss_function = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}\n", flush=True)
    model.train()
    
    train_loss = 0
    for i, (image, mask) in enumerate(train_loader):
        output = model(apply_mask_on_channel(image, mask, placeholder))
        inverse_mask = 1 - mask
        loss_val = loss_function(apply_mask_on_channel(output, inverse_mask, placeholder), apply_mask_on_channel(image, inverse_mask, placeholder)) / th.sum(inverse_mask)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        train_loss += loss_val.item()
    
    train_losses.append(train_loss / len(train_loader))
    
    with th.no_grad():
        model.eval()
        test_loss = 0
        for i, (image, mask) in enumerate(test_loader):
            output = model(apply_mask_on_channel(image, mask, placeholder))
            inverse_mask = 1 - mask
            loss_val = loss_function(apply_mask_on_channel(output, inverse_mask, placeholder), apply_mask_on_channel(image, inverse_mask, placeholder)) / th.sum(inverse_mask)
            test_loss += loss_val.item()
        
        test_losses.append(test_loss / len(test_loader))

elapsed_time = time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Save the model
th.save(model.state_dict(), weights_path)

# Save the train losses to a txt file
with open(results_path, 'w') as f:
    f.write("Elapsed time [s]:\n")
    f.write(f"{elapsed_time}\n\n")
    f.write("Train losses\n")
    for i, loss_function in enumerate(train_losses):
        f.write(f"{loss_function}\t")
    f.write("\n\n")
    f.write("Test losses\n")
    for i, loss_function in enumerate(test_losses):
        f.write(f"{loss_function}\t")
    f.write("\n\n")
    
        
