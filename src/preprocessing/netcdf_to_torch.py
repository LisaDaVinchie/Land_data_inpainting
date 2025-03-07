import xarray as xr
import torch as th
import os
import sys
import time
from pathlib import Path
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.import_params_json import load_config

start_time = time.time()

parser = argparse.ArgumentParser(description='Convert NetCDF to Torch')
parser.add_argument('--paths', type=Path, help='File containing paths', required=True)

args = parser.parse_args()
paths_file = Path(args.paths)

# Load the paths
raw_data_dir: Path = None
processed_data_dir: Path = None
processed_data_ext: str = None
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
# extract year, month and day from the name

print("\nLoading the data\n")
data = xr.open_dataset(raw_data_path, engine="h5netcdf")


# Get the keys
print("Retrieving keys\n")
keys = list(data.keys())
# keys = list(data.variables.keys())
print(f"Keys: {keys}")

print("")

n_lats = data["or_latitude"].shape[1]
n_lons = data["or_longitude"].shape[2]

print(f"Number of lats: {n_lats}")
print(f"Number of lons: {n_lons}")
del data

# Assuming all the files have the same keys and shape

print(f"Processing {len(raw_data_paths)} files")
for file in raw_data_paths:
    file_name = file.name
    print(f"Processing {file_name}\n")
    year = file_name[0:4]
    month = file_name[4:6]
    day = file_name[6:8]
    processed_data_path = processed_data_dir / f"{year}_{month}_{day}{processed_data_ext}"
    output_tensor = th.zeros(len(keys), n_lats, n_lons)
    try:
        data = xr.open_dataset(file, engine="h5netcdf")

        for i, key in enumerate(keys):
            if key != "crs":
                output_tensor[i, :, :] = th.tensor(data[key].values).unsqueeze(1)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        continue

    th.save(output_tensor, processed_data_path)
    
elapsed_time = time.time() - start_time

print(f"Elapsed time: {elapsed_time}")