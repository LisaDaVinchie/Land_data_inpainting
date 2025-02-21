import xarray as xr
import torch as th
from pathlib import Path
import argparse
from utils.import_params_json import load_config

parser = argparse.ArgumentParser(description='Convert NetCDF to Torch')

parser.add_argument('--paths', type=Path, help='File containing paths', required=True)

args = parser.parse_args()
paths_file = Path(args.paths)

# Load the paths
raw_data_path: Path = None
processed_data_path: Path = None
config = load_config(paths_file, ["data"])
locals().update(config["data"])

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
if not processed_data_path.parent.exists():
    processed_data_path.parent.mkdir(parents=True)
th.save(output_tensor, processed_data_path)