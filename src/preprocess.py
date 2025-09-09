import numpy as np
import xarray as xr
from pathlib import Path

def find_land_sea_mask(dataset: np.ndarray, perc: float):
    """Find the land sea mask, as the pixels that have valid data in more than perc percentage of the dataset

    Args:
        dataset (np.ndarray): The input dataset, shape (N, H, W)
        perc (float): The percentage threshold, 0 < perc < 1

    Returns:
        np.ndarray: The land sea mask as a boolean array of shape (H, W)
    """
    N = dataset.shape[0]
    threshold = int(N * perc)
    mask = ~np.isnan(dataset)

    sum_mask = np.sum(mask, axis=0)

    return sum_mask > threshold

def find_cloud_mask(dataset: np.ndarray, land_sea_mask: np.ndarray):
    """Find the cloud mask, as the pixels that have valid data and temperature below a threshold

    Args:
        dataset (np.ndarray): The input dataset, shape (N, H, W)
        land_sea_mask (np.ndarray): The land sea mask, shape (H, W)

    Returns:
        np.ndarray: The cloud mask as a boolean array of shape (N, H, W)
    """
    n_land = ~np.sum(land_sea_mask)
    cloud_mask = land_sea_mask & np.isnan(dataset)

    n_cloud = [np.sum(~cloud_mask[i]) for i in range(cloud_mask.shape[0])]

    return cloud_mask[n_cloud > n_land]
    
PERC = 0.05 # 5%
test = False
if test:
    dataset_path = Path('./data/minimal_datasets/dataset_1_test.nc')
    output_path = Path('./data/minimal_datasets/dataset_proc_1_test.nc')
else:
    dataset_path = Path('./data/minimal_datasets/dataset_1.nc')
    output_path = Path('./data/minimal_datasets/dataset_proc_1.nc')

time_win = 3  # Not used in this script but may be relevant for context
n_days = time_win // 2

if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

ds = xr.load_dataset(dataset_path)
print("Dataset loaded, keys are:", list(ds.data_vars))

ds_sst = ds['sst'].values
meanval = np.nanmean(ds_sst)
stdval = np.nanstd(ds_sst)

nan_mask = ~np.isnan(ds_sst)
sea_mask = find_land_sea_mask(ds_sst, PERC)
mask = find_cloud_mask(ds_sst, sea_mask)
cloud_mask = np.ones((mask.shape[0], time_win + 4, mask.shape[1], mask.shape[2]), dtype=bool)
cloud_mask[:, 1:2, :, :] = mask[:, np.newaxis, :, :]
print(f"Data normalized: mean={meanval}, std={stdval}")
print(f"cloud mask shape: {cloud_mask.shape}\n")

ds = ds.sortby('time') # Ensure time dimension is sorted
print("Dataset sorted by time.")

# Scale lat/lon to [-1, 1]
def minmax_scale(coord):
    """Scale coordinates to [-1, 1]."""
    minval = coord.min()
    maxval = coord.max()
    return 2 * (coord - minval) / (maxval - minval) - 1

ds['lat'] = minmax_scale(ds['lat'])
ds['lon'] = minmax_scale(ds['lon'])

lats = ds['lat'].values.repeat(ds.sizes['lon'], axis=0).reshape(ds.sizes['lat'], ds.sizes['lon'])
lons = ds['lon'].values.repeat(ds.sizes['lat'], axis=0).reshape(ds.sizes['lon'], ds.sizes['lat']).T

print("Latitude and longitude scaled to [-1, 1].\n")


def time_encode(t: np.datetime64, period: int = 365.25, date_start: np.datetime64 = np.datetime64('1970-01-01')) -> np.ndarray:
    """Encode time as a cyclical feature."""
    n_days = (t - date_start) / np.timedelta64(1, 'D')
    sin_time = np.sin(2 * np.pi * n_days / period)
    cos_time = np.cos(2 * np.pi * n_days / period)
    return sin_time, cos_time

H = ds.sst.shape[1]
W = ds.sst.shape[2]

N_sea = np.sum(sea_mask)
thresh = int(N_sea * 0.05) # At least 5% of sea pixels must be valid
print(f"Number of sea pixels: {N_sea}, threshold for valid data: {thresh}\n")

print("Adding time windows and encodings...")

sst_arr = []
time_arr = []
nan_mask = []
for time in ds.time.values:
    arr = np.zeros((1, time_win + 4, H, W), dtype=np.float32)
    
    sst = ds['sst'].sel(time=slice(time - np.timedelta64(n_days, 'D'), time + np.timedelta64(n_days, 'D'))).values
    if sst.shape[0] != time_win:
        print(f"Skipping time {time} due to insufficient data for time window.\n")
        
        continue
    
    if np.sum(~np.isnan(sst)) < thresh:
        print(f"Skipping time {time} due to insufficient valid data.\n")
        continue

    arr[0, 0:time_win, :, :] = sst

    # Add time encodings
    sin_time, cos_time = time_encode(time)
    arr[0, -4, :, :] = sin_time * np.ones((H, W))
    arr[0, -3, :, :] = cos_time * np.ones((H, W))
    arr[0, -2, :, :] = lats
    arr[0, -1, :, :] = lons
    
    sst_arr.append(arr)
    time_arr.append(time)
    nan_mask.append(~np.isnan(arr))
print("Time windows and encodings added.\n")

sst_arr = np.concatenate(sst_arr, axis=0)
nan_mask = np.concatenate(nan_mask, axis=0)

new_ds = xr.Dataset(
    {
        'sst': (['time', 'channels', 'lat', 'lon'], sst_arr),
        'nan_mask': (['time', 'channels', 'lat', 'lon'], nan_mask),
        'mask': (['n_masks', 'channels', 'lat', 'lon'], cloud_mask),
        'meanval': ((), meanval),
        'stdval': ((), stdval),
        'land_sea_mask': (['lat', 'lon'], sea_mask)
    },
    coords={
        'time': time_arr,
        'lat': ds['lat'],
        'lon': ds['lon'],
        'channels': np.arange(time_win + 4)
    }
)

print("final sst shape is:", new_ds['sst'].shape)
new_ds['sst'] = (new_ds['sst'] - meanval) / stdval  # Normalize
new_ds['sst'] = new_ds['sst'].fillna(-300.0)  # Fill NaNs with -300 for model compatibility
new_ds.to_netcdf(output_path)
print(f"Processed dataset saved to: {output_path}")