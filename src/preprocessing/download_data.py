import requests
import argparse
import os
import sys
from time import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.import_params_json import load_config

start_time = time()
parser = argparse.ArgumentParser(description = "Download data from Copernicus.")
parser.add_argument("--paths", type = Path, help = "The json file containing the paths.")
parser.add_argument("--params", type = Path, help = "The json file containing the parameters.")

args = parser.parse_args()
paths_file_path = args.paths
params_file_path = args.params

raw_data_dir: Path = None
paths = load_config(paths_file_path, ["data"])
locals().update(paths["data"])
raw_data_dir = Path(raw_data_dir)
del paths

if not raw_data_dir.exists():
    raise FileNotFoundError(f"Directory {raw_data_dir} does not exist.")

dataset_name: str = None
years: str = None
months: str = None
days: str = None
config = load_config(params_file_path, ["dataset"])
locals().update(config["dataset"])
del config


separator = "%2F"
copericus_products_url = "https://data.marine.copernicus.eu/product/"
dataset_path = "SST_GLO_SST_L3S_NRT_OBSERVATIONS_010_010/files?path=SST_GLO_SST_L3S_NRT_OBSERVATIONS_010_010"
online_file_basename = "000000-IFR-L3S_GHRSST-SSTfnd-ODYSSEA-GLOB_010-v02.1-fv01.0"
extension = ".nc"

session = requests.Session()
i = 0
for year in years:
    year = str(year)
    year_url = copericus_products_url + dataset_path + separator + dataset_name + separator + year + separator
    for month in months:
        month = str(month).zfill(2)
        month_url = year_url + month + separator
        for day in days:
            if day == 31 and month in ["04", "06", "09", "11"]:
                continue
            if day == 30 and month == "02":
                continue
            day = str(day).zfill(2)
            url = month_url + year + month + day + online_file_basename + extension
            file_name = Path(year + "_" + month + "_" + day + extension)
            
            response = session.get(url)
            
            if response.status_code == 200:
                with open(raw_data_dir / file_name, "wb") as f:
                    f.write(response.content)
                i = i + 1
                
            else:
                print(f"Could not download file {file_name}, status code: {response.status_code}")

elapsed_time = time() - start_time
print(f"\n\nDownloaded {i} files in {elapsed_time} s.")