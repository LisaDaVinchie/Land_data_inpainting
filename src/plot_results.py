import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

def read_results(results_path: Path) -> tuple:
    with open(results_path, 'r') as f:
        lines = f.readlines()
    
    train_losses = None
    test_losses = None

    for i, line in enumerate(lines):
        if line.strip() == "Train losses":
        # Next line contains the values
            values = lines[i+1].strip().split('\t')
            train_losses = np.array([float(v) for v in values if v])
        elif line.strip() == "Test losses":
        # Next line contains the values
            values = lines[i+1].strip().split('\t')
            test_losses = np.array([float(v) for v in values if v])
        
    if train_losses is None or test_losses is None:
        raise ValueError("Could not find train or test losses in the results file.")
    return train_losses, test_losses

def main():
    parser = argparse.ArgumentParser(description='Plot results from training')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    args = parser.parse_args()

    paths_path = Path(args.paths)

    with open(paths_path, 'r') as f:
        paths = json.load(f)
    results_path = Path(paths["results"]["current_results_path"])
    figs_path = Path(paths["results"]["figs_path"])

    for path in [results_path, figs_path.parent]:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

    train_losses, test_losses = read_results(results_path)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xticks(range(len(train_losses)), range(1, len(train_losses) + 1))

    plt.savefig(figs_path)
    
if __name__ == "__main__":
    main()

