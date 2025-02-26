import xarray as xr
import torch as th
import os
import sys
from pathlib import Path
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.import_params_json import load_config

parser = argparse.ArgumentParser(description='Convert NetCDF to Torch')

parser.add_argument('--paths', type=Path, help='File containing paths', required=True)

args = parser.parse_args()
paths_file = Path(args.paths)

# Load the paths
raw_data_dir: Path = None
processed_data_dir: Path = None
config = load_config(paths_file, ["data"])
locals().update(config["data"])
raw_data_dir = Path(raw_data_dir)
processed_data_dir = Path(processed_data_dir)


if not raw_data_dir.exists():
    raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")
if not processed_data_dir.exists():
    raise FileNotFoundError(f"Directory {processed_data_dir} does not exist.")

raw_data_paths = list(raw_data_dir.glob("*.nc"))

raw_data_path = raw_data_paths[0]
print(f"Processing {raw_data_path}")
processed_data_path = processed_data_dir / raw_data_path.name.replace(".nc", ".pth")

print("\nLoading the data\n")
data = xr.open_dataset(raw_data_path)


# Get the keys
print("Retrieving keys\n")
keys = list(data.keys())

n_lats = data["latitude"].shape[0]
n_lons = data["longitude"].shape[0]
n_days = data["time"].shape[0]

print(f"Number of lats: {n_lats}")
print(f"Number of lons: {n_lons}")
print(f"Number of days: {n_days}")

output_tensor = th.zeros(n_days, len(keys), n_lats, n_lons)

for i, key in enumerate(keys):
    print(f"Processing {key}")
    output_tensor[:, i, :, :] = th.tensor(data[key].values)

# Save the tensor
print(f"Saving the tensor to {processed_data_path}")
th.save(output_tensor, processed_data_path)