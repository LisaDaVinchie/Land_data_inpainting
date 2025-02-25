import torch as th
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from torch.utils.data import random_split
from pathlib import Path
import argparse
from models import simple_conv, conv_maxpool
from time import time

from mask_data import SquareMask, mask_image
from mask_data import mask_image, SquareMask
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
paths = load_config(paths_path, ["results"])
locals().update(paths["results"])

results_path = Path(results_path)
figs_path = Path(figs_path)
weights_path = Path(weights_path)

if not results_path.parent.exists():
    print("Results folder does not exist.", flush = True)
    raise FileNotFoundError(f"Results folder does not exist: {results_path.parent}")

if not figs_path.parent.exists():
    print("Figures folder does not exist.", flush = True)
    raise FileNotFoundError(f"Figures folder does not exist: {figs_path.parent}")

if not weights_path.parent.exists():
    print("Weights folder does not exist.", flush = True)
    raise FileNotFoundError(f"Weights folder does not exist: {weights_path.parent}")

# Load the train parameters
train_perc: int = None
epochs: int = None
batch_size: int = None
learning_rate: float = None
optimizer_kind: str = None
loss_kind: str = None
model_kind: str = None
config = load_config(params_path, ["training"])
locals().update(config["training"])

# Import dataset
dataset = th.rand(100, 13, 10, 10)
mask_class = SquareMask(params_path)
masks = []
for i in range(len(dataset)):
    masks.append(mask_class.mask())

masks = th.stack(masks)

print(f"Dataset shape: {dataset.shape}, Masks shape: {masks.shape}\n")

# Split the dataset into train and test sets
train_size = int(train_perc * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_set = CustomDataset(train_dataset, masks)
test_set = CustomDataset(test_dataset, masks)

# Create DataLoader for train and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

model = simple_conv(params_path)
loss_function = th.nn.MSELoss()
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}\n", flush=True)
    model.train()
    
    train_loss = 0
    for i, (image, mask) in enumerate(train_loader):
        output = model(mask_image(image, mask, 0))
        loss_val = loss_function(mask_image(output, 1 - mask, 0), mask_image(image, 1 - mask, 0)) / th.sum(1 - mask)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        train_loss += loss_val.item()
    
    train_losses.append(train_loss / len(train_loader))
    
    with th.no_grad():
        model.eval()
        test_loss = 0
        for i, (image, mask) in enumerate(test_loader):
            output = model(mask_image(image, mask, 0))
            loss_val = loss_function(mask_image(output, 1 - mask, 0), mask_image(image, 1- mask, 0)) / th.sum(1 - mask)
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
    
        
