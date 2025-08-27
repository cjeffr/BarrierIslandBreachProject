# --- Build a single NetCDF from 6 gauge .pkl files ---
# Uses xarray + NetCDF4 (portable, compressed, self-describing).
# Dim names: time, gauge, simulation

import pandas as pd
import numpy as np
import xarray as xr
import re
from pathlib import Path
import os
path = '/home/catherinej/BarrierIslandBreachProject/data/processed/'
# -------------------- USER INPUTS --------------------
pkl_paths = [
    os.path.join(path, "west_gauge10084.pkl.gz"),
    os.path.join(path, "west_gauge10082.pkl.gz"),
    os.path.join(path, "central_gauge10045.pkl.gz"),
    os.path.join(path, "central_gauge10133.pkl.gz"),
    os.path.join(path, "east_gauge10011.pkl.gz"),
    os.path.join(path, "east_gauge10119.pkl.gz")
]
gauge_ids = ["a","b","c","d","e","f"]  # labels for your 6 gauges

output_nc = "gauges_ensemble.nc"

# Optional: parse a "category" from each simulation name.
# Customize this to your naming convention. Example below:
def parse_category(col_name: str) -> str:
    name = str(col_name)
    if name == "no_breach":
        return "no_breach"
    # check 'dw' first so it doesn't get caught by 'w' or 'd'
    if "dw" in name:
        return "dw"
    # your original filters excluded the other letter
    if ("w" in name) and ("d" not in name):
        return "w"
    if ("d" in name) and ("w" not in name):
        return "d"
    if "loc" in name:
        return "loc"
    return "uncategorized"


# -------------------- LOAD & CHECK --------------------
dfs = []
for p in pkl_paths:
    df = pd.read_pickle(p)
    df = df.copy()
    df.index.name = "time"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    dfs.append(df)

# Ensure all gauges have identical simulation columns (order too)
base_cols = list(dfs[0].columns)
for i, df in enumerate(dfs[1:], start=2):
    if list(df.columns) != base_cols:
        raise ValueError(f"Columns in file {i} differ from file 1. "
                         "Since you said they’re identical, this is unexpected.")

# Align on a common time index (inner join)
time_index = dfs[0].index
for df in dfs[1:]:
    time_index = time_index.intersection(df.index)
if len(time_index) == 0:
    raise ValueError("No overlapping time steps across files.")

# Reindex each DF to the common time
dfs = [df.loc[time_index] for df in dfs]

# Convert to 3D array: [time, gauge, simulation]
# Use float32 to reduce size (change to float64 if you truly need it)
time = time_index.to_numpy(dtype=np.float64)     # seconds since landfall
simulation_names = base_cols                     # list of column headers
data_stack = np.stack([df.to_numpy(dtype=np.float32) for df in dfs], axis=1)  # [time, gauge, simulation]

# Build category (one per simulation; same for all gauges)
categories = np.array([parse_category(c) for c in simulation_names], dtype=object)

# -------------------- XARRAY SOLUTION --------------------
ds = xr.Dataset(
    data_vars={
        "water_level_raw": (("time", "gauge", "simulation"), data_stack),
    },
    coords={
        "time": time,
        "gauge": gauge_ids,
        "simulation": np.arange(len(simulation_names), dtype=np.int32),
        "simulation_name": ("simulation", np.array(simulation_names, dtype=object)),
        "category": ("simulation", categories),
    },
)

# -------------------- SUMMARY STATS (all simulations) --------------------
ds["water_level_mean"]   = ds["water_level_raw"].mean(dim="simulation", skipna=True)
ds["water_level_median"] = ds["water_level_raw"].median(dim="simulation", skipna=True)
ds["water_level_p05"]    = ds["water_level_raw"].quantile(0.05, dim="simulation", skipna=True)
ds["water_level_p95"]    = ds["water_level_raw"].quantile(0.95, dim="simulation", skipna=True)

# -------------------- (Optional) STATS BY CATEGORY --------------------
# Produces variables of shape [time, gauge, category]
# Comment out this block if you don't need per-category summaries.
cats = np.unique(categories)
def _cat_mask(cat):
    return xr.DataArray((ds["category"].values == cat), dims=("simulation",))

bycat = {}
for cat in cats:
    mask = _cat_mask(cat)
    # Mask simulations that belong to this category
    sub = ds["water_level_raw"].where(mask, drop=True)
    # If a category has no members (shouldn't happen), skip
    if sub.sizes.get("simulation", 0) == 0:
        continue
    bycat.setdefault("mean",   []).append(sub.mean(dim="simulation", skipna=True).assign_coords(category=cat))
    bycat.setdefault("median", []).append(sub.median(dim="simulation", skipna=True).assign_coords(category=cat))
    bycat.setdefault("p05",    []).append(sub.quantile(0.05, dim="simulation", skipna=True).assign_coords(category=cat))
    bycat.setdefault("p95",    []).append(sub.quantile(0.95, dim="simulation", skipna=True).assign_coords(category=cat))

if bycat:
    ds["water_level_mean_bycat"]   = xr.concat(bycat["mean"],   dim="category")
    ds["water_level_median_bycat"] = xr.concat(bycat["median"], dim="category")
    ds["water_level_p05_bycat"]    = xr.concat(bycat["p05"],    dim="category")
    ds["water_level_p95_bycat"]    = xr.concat(bycat["p95"],    dim="category")

# -------------------- METADATA --------------------
ds.attrs.update({
    "title": "1938 Hurricane gauge ensemble",
    "summary": "Ensemble simulations at six gauges; time is seconds relative to landfall.",
    "creator_name": "Catherine R Jeffries",
    "institution": "Virginia Tech, Department of Geosciences",
    "license": "CC-BY-4.0",
    "conventions": "CF-1.10",
    "landfall_reference": "time=0 corresponds to hurricane landfall at 09/21/1938 19:45 UTC",
    "history": "Created from 6 pickle files; compressed and chunked for archival use.",
})
ds["time"].attrs.update({
    "standard_name": "time",
    "units": "seconds since landfall",
    "axis": "T",
})
ds["gauge"].attrs.update({
    "long_name": "gauge identifier",
})
ds["simulation"].attrs.update({
    "long_name": "simulation index",
})
ds["simulation_name"].attrs.update({
    "long_name": "original simulation header/name",
})
ds["category"].attrs.update({
    "long_name": "simulation category parsed from simulation_name",
})

# Variable attributes (adjust units/standard_name to your quantity)
for v in ["water_level_raw","water_level_mean","water_level_median",
          "water_level_p05","water_level_p95",
          "water_level_mean_bycat","water_level_median_bycat",
          "water_level_p05_bycat","water_level_p95_bycat"]:
    if v in ds:
        ds[v].attrs.setdefault("long_name", "water level")
        ds[v].attrs.setdefault("units", "m")  # <-- change to your units
        if v.endswith("_bycat"):
            ds[v].attrs["description"] = "Summary across simulations grouped by category"

# numeric compression settings
enc_f32 = dict(zlib=True, complevel=5, shuffle=True, dtype="float32")

def target_chunk_for_dim(dim_name, dim_size):
    if dim_name == "time":
        return min(dim_size, 4096)
    if dim_name in ("simulation", "quantile"):
        return min(dim_size, 64)
    if dim_name in ("category",):
        return min(dim_size, 8)
    if dim_name == "gauge":
        return min(dim_size, 1)
    return min(dim_size, 64)

encoding = {}
numeric_vars = [
    "water_level_raw",
    "water_level_mean", "water_level_median",
    "water_level_p05", "water_level_p95",
    "water_level_mean_bycat", "water_level_median_bycat",
    "water_level_p05_bycat", "water_level_p95_bycat",
]
for v in numeric_vars:
    if v in ds:
        chunksizes = tuple(
            target_chunk_for_dim(d, ds.sizes[d]) for d in ds[v].dims
        )
        encoding[v] = {**enc_f32, "chunksizes": chunksizes}

# IMPORTANT: do not add any encoding entries for string coords/vars:
#   - simulation_name (string), category (string), gauge_region (string)
# Writing:
ds.to_netcdf(output_nc, format="NETCDF4", encoding=encoding)

