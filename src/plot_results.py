import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from utils import read_results

def main():
    parser = argparse.ArgumentParser(description='Plot results from training')
    parser.add_argument('--respath', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--figdir', type=Path, help='Path to the JSON file with data paths')
    args = parser.parse_args()

    results_path = Path(args.respath)
    figs_dir = Path(args.figdir)
    
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file {results_path} does not exist.")
    
    if not figs_dir.is_dir():
        raise FileNotFoundError(f"Figures directory {figs_dir} does not exist.")
    
    fig_path = figs_dir / str(results_path.stem + ".png")

    train_losses, test_losses = read_results(results_path)

    print("Plotting results from", results_path, flush=True)
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xticks(range(len(train_losses)), range(1, len(train_losses) + 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fig_path)
    print("Plot saved to", fig_path, flush=True)
    
if __name__ == "__main__":
    main()

