import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from time import time
import json
import psutil
import os

from CustomDataset import create_dataloaders
from models import initialize_model_and_dataset_kind
from preprocessing.mask_data import mask_inversemask_image
from losses import get_loss_function

def main():
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')
    parser.add_argument('--dataset_idx', type=int, help='Index of the dataset to use', required=False)

    args = parser.parse_args()
    
    start_time = time()

    params_path = args.params
    paths_path = args.paths
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    train_perc = params["training"]["train_perc"]
    epochs = params["training"]["epochs"]
    batch_size = params["training"]["batch_size"]
    learning_rate = params["training"]["learning_rate"]
    optimizer_kind = params["training"]["optimizer_kind"]
    lr_scheduler = params["training"]["lr_scheduler"]
    loss_kind = params["training"]["loss_kind"]
    model_kind = params["training"]["model_kind"]
    placeholder = params["training"]["placeholder"]
    nan_placeholder = params["dataset"]["nan_placeholder"]
    print("Parameters imported\n", flush = True)
    
    with open(paths_path, 'r') as f:
        paths = json.load(f)
    
    results_path = Path(paths["results"]["results_path"])
    weights_path = Path(paths["results"]["weights_path"])
    current_minimal_dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    
    track_memory("Before loading model")
    model, dataset_kind = initialize_model_and_dataset_kind(params_path, model_kind)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = model.to(device)
    track_memory("After loading model")
    print("Model initialized\n", flush = True)
    
    dataset_path = current_minimal_dataset_path
    idx = args.dataset_idx
    dataset_path = change_dataset_idx(idx, dataset_path)
    print("Using dataset path:", dataset_path, flush = True)
    
    validate_paths([dataset_path, results_path.parent, weights_path.parent])
    print("\nPaths imported\n", flush = True)

    dataset_start_time = time()
    track_memory("Before loading dataset")
    dataset = th.load(dataset_path)
    track_memory("After loading dataset")
    print(f"Dataset loaded in {time() - dataset_start_time:.2f} seconds", flush = True)

    track_memory("Before creating dataloaders")
    train_loader, test_loader = create_dataloaders(dataset, train_perc, batch_size)
    del dataset
    print("Dataloaders created\n", flush = True)
    track_memory("After creating dataloaders")
    
    loss_function = get_loss_function(loss_kind, nan_placeholder)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if lr_scheduler == "step":
        step_size = int(params["training"]["step_size"])
        gamma = float(params["training"]["gamma"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    training_time = time()
    track_memory("Before training")
    if dataset_kind == "extended":
        model, train_losses, test_losses = train_loop_extended(epochs, placeholder, model, device, train_loader, test_loader, loss_function, optimizer)
    elif dataset_kind == "minimal":
        model, train_losses, test_losses = train_loop_minimal(epochs, model, device, train_loader, test_loader, loss_function, optimizer)
    else:
        raise ValueError(f"Dataset kind {dataset_kind} not recognized")
    track_memory("After training")
    print(f"Training completed in {time() - training_time:.2f} seconds", flush = True)

    # Save the model
    th.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}\n", flush = True)

    with open(params_path, 'r') as f:
        params = json.load(f)
        
    json_str = json.dumps(params, indent=4)[1: -1]
    
    elapsed_time = time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Save the train losses to a txt file
    with open(results_path, 'w') as f:
        f.write("Elapsed time [s]:\n")
        f.write(f"{elapsed_time}\n\n")
        f.write("Used dataset:\n")
        f.write(f"{dataset_path.relative_to(dataset_path.parent.parent)}\n\n")
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
    
    print(f"Results saved to {results_path}\n", flush = True)
        
def track_memory(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / 1e6  # Convert to MB
    print(f"[Memory] {stage}: {mem_info:.2f} MB\n", flush=True)
    
def change_dataset_idx(dataset_idx: int, dataset_path: Path) -> Path:
    """Take the latest dataset path by default, or change the dataset index if specified.

    Args:
        dataset_idx (int): desired dataset index
        dataset_path (Path): path to the dataset

    Returns:
        Path: path to the dataset with the desired index
    """
    
    if dataset_idx is None:
        return dataset_path
    
    dataset_basepath = dataset_path.stem.split("_")[0]
    dataset_ext = dataset_path.suffix
    dataset_path = dataset_path.parent / f"{dataset_basepath}_{dataset_idx}{dataset_ext}"
    return dataset_path

def validate_paths(paths: list):
    """Check if the paths exist.

    Raises:
        FileNotFoundError: if any of the paths do not exist, raise an error
    """
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        
def train_loop_extended(epochs: int, placeholder: float, model: th.nn.Module, device, train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module, optimizer: optim.Optimizer):
    """Training loop for the extended dataset."""
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}\n", flush=True)
        model.train()
        
        train_loss = 0
        for i, (image, mask) in enumerate(train_loader):
            masked_image, _ = mask_inversemask_image(image, mask, placeholder)
            masked_image = masked_image.to(device)
            output = model(masked_image)
            loss_val = loss_function(output, image, mask)
            train_loss += loss_val.item()
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        
        train_losses.append(train_loss / len(train_loader))
        
        with th.no_grad():
            model.eval()
            test_loss = 0
            for i, (image, mask) in enumerate(test_loader):
                masked_image, _ = mask_inversemask_image(image, mask, placeholder)
                output = model(masked_image)
                loss_val = loss_function(output, image, mask)
                test_loss += loss_val.item()
            
            test_losses.append(test_loss / len(test_loader))
    return model, train_losses, test_losses

def train_loop_minimal(epochs: int, model: th.nn.Module, device, train_loader: DataLoader, test_loader: DataLoader, loss_function: nn.Module, optimizer: optim.Optimizer):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}\n", flush=True)
        
        model.train()
        train_loss = 0
        for (images, masks) in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            output, output_masks = model(images, masks)
            
            loss_val = loss_function(output, images, masks)
            train_loss += loss_val.item()

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        
        train_losses.append(train_loss / len(train_loader))
        
        with th.no_grad():
            model.eval()
            test_loss = 0
            for (images, masks) in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                output, output_masks = model(images, masks)
                loss_val = loss_function(output, images, masks)
                test_loss += loss_val.item()
            
            test_losses.append(test_loss / len(test_loader))
    return model, train_losses, test_losses
        
if __name__ == "__main__":
    main()