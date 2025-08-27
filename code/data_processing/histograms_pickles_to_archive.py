#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exports figure-ready archives from pickles with two known layouts:

POINTS files (*_fgmax_pts.pkl.gz or *_fgmax_points.pkl.gz):
  - columns: MultiIndex (lat, lon), one per location
  - rows   : metric names (including 'lat' and 'lon' as rows)
  - we choose ONE location column deterministically and take its values per metric

REGIONAL files (*_fg_max.pkl.gz):
  - columns: include 'lat', 'lon', and many metric columns
  - rows   : contain exactly one non-NaN per metric column
  - we take the first non-NaN per metric COLUMN

Outputs:
  - histogram_archive.nc      (authoritative NetCDF)
  - histogram_archive_long.csv
  - ./csv/{figure|scope|region|kind}=*.csv

Usage:
  python export_archive.py
"""

from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import xarray as xr

# -----------------------------
# CONFIG — edit paths/names here
# -----------------------------
BASE = '/home/catherinej/BarrierIslandBreachProject/data/processed/processed'

DATASETS = [
    {
        "figure": "Fig5",        # POINTS (per your corrected mapping)
        "scope":  "point",
        "paths": {
            "west":    os.path.join(BASE, "west_fgmax_points.pkl.gz"),
            "central": os.path.join(BASE, "central_fgmax_pts.pkl.gz"),
            "east":    os.path.join(BASE, "east_fgmax_pts.pkl.gz"),
        },
    },
    {
        "figure": "Fig6",        # REGIONAL
        "scope":  "region",
        "paths": {
            "west":    os.path.join(BASE, "west_fg_max.pkl.gz"),
            "central": os.path.join(BASE, "central_fg_max.pkl.gz"),
            "east":    os.path.join(BASE, "east_fg_max.pkl.gz"),
        },
    },
]

# If a POINTS file has multiple location columns, which one do we use?
# Options: "middle" (default), "first", "last", or an explicit integer index.
POINTS_COLUMN_SELECTION = "middle"

# Outputs
OUT_NC      = "histogram_archive.nc"
OUT_CSV     = "histogram_archive_long.csv"
OUT_CSV_DIR = Path("csv")

# Bucketing (plot groups)
BUCKETS = {
    "width_depth": r"(?i)(^|_)dw(?:_|$|\d+)",
    "width":       r"(?i)(^|_)w(?!est)(?:_|$|\d+)",
    "depth":       r"(?i)(^|_)d(?:_|$|\d+)",
    "locations":   r"(?i)(^|_)loc(?:_|$|\d+)",
    "east":        r"(?i)^(east)(?:_|$|\d+)",
    "west":        r"(?i)^(west)(?:_|$|\d+)",
}

# -----------------------------
# Helpers
# -----------------------------

def _bucket_kind(name: str) -> str | None:
    if not isinstance(name, str):
        name = str(name)
    for kind, rx in BUCKETS.items():
        if re.search(rx, name):
            return kind
    return None

def _pick_points_column(cols) -> int:
    """Choose which location column to use for POINTS files, deterministically."""
    n = len(cols)
    if n == 0:
        raise ValueError("POINTS file has zero columns.")
    if isinstance(POINTS_COLUMN_SELECTION, int):
        idx = POINTS_COLUMN_SELECTION
    elif POINTS_COLUMN_SELECTION == "first":
        idx = 0
    elif POINTS_COLUMN_SELECTION == "last":
        idx = n - 1
    else:  # "middle" (default)
        idx = n // 2
    if not (0 <= idx < n):
        raise IndexError(f"POINTS_COLUMN_SELECTION resolved to {idx}, but ncols={n}.")
    return idx

def load_metrics_series(df: pd.DataFrame, expect: str) -> pd.Series:
    """
    Return a Series keyed by metric name with one numeric value per metric.
    expect: "points" or "regional"
    """
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    # Normalize dtypes / labels
    df = df.copy()
    # Note: columns for POINTS are a MultiIndex with names ('lat','lon')
    # We will not stringify columns first; we inspect structure.

    if expect == "regional":
        # Columns contain 'lat','lon' and many metric columns; rows have exactly one non-NaN per metric.
        # -> Take first valid per metric column.
        # Ensure string columns so we can drop by name
        df.columns = df.columns.map(str)
        metric_cols = [c for c in df.columns if c not in ("lat", "lon")]
        if not metric_cols:
            raise ValueError("REGIONAL file missing metric columns (found only lat/lon?).")
        num = df[metric_cols].apply(pd.to_numeric, errors="coerce")
        out = {}
        for c in num.columns:
            s = num[c]
            idx = s.first_valid_index()
            if idx is not None and pd.notna(s.loc[idx]):
                out[c] = float(s.loc[idx])
        return pd.Series(out, dtype="float64")

    elif expect == "points":
        # Columns: MultiIndex of (lat, lon); Rows: metrics (including rows named 'lat' and 'lon').
        # We choose one location column, then read values from that column, dropping the 'lat'/'lon' rows.
        if not isinstance(df.columns, pd.MultiIndex) or set(df.columns.names) != {"lat", "lon"}:
            # Try to convert first column to index if it contains 'lat'/'lon' etc. (rare)
            # but your sample shows the canonical MultiIndex columns, so raise by default:
            raise ValueError("POINTS file not recognized: expected columns MultiIndex with names ('lat','lon').")

        col_idx = _pick_points_column(df.columns)
        col_key = df.columns[col_idx]

        # Pull that column as a Series indexed by metric names
        s = df.iloc[:, col_idx]
        # s index are metric names; drop the lat/lon metric rows
        idx_names = s.index.map(str)
        mask = ~idx_names.isin(["lat", "lon"])
        s = pd.to_numeric(s[mask], errors="coerce").dropna()

        # Make sure the result’s index are clean strings
        s.index = s.index.map(str)
        return s.astype("float64")

    else:
        raise ValueError(f"expect must be 'points' or 'regional', got: {expect}")

# -----------------------------
# Main export
# -----------------------------

def main():
    rows = []
    for group in DATASETS:
        figure = group["figure"]
        scope  = group["scope"]
        for region, path in group["paths"].items():
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Missing pickle for {figure}/{scope}/{region}: {p}")
            df = pd.read_pickle(p)

            expect = "points" if scope == "point" else "regional"
            series = load_metrics_series(df, expect=expect)

            # Tidy frame
            part = (
                series.rename("value")
                      .reset_index()
                      .rename(columns={"index": "source_name"})
            )
            part["region"] = region
            part["figure"] = figure
            part["scope"]  = scope

            # bucket
            part["kind"] = part["source_name"].apply(_bucket_kind)
            rows.append(part[["figure","scope","region","kind","source_name","value"]])

    long = (
        pd.concat(rows, ignore_index=True)
          .dropna(subset=["value"])
          .sort_values(["figure","scope","region","kind","source_name"])
          .reset_index(drop=True)
    )

    # Write CSV
    long.to_csv(OUT_CSV, index=False)

    # Convenience CSVs
    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    for col in ["figure", "scope", "region", "kind"]:
        for val in sorted(long[col].dropna().unique()):
            long[long[col] == val].to_csv(OUT_CSV_DIR / f"{col}={val}.csv", index=False)

    # NetCDF (discrete sampling style; coords per observation)
    ds = xr.Dataset(
        data_vars=dict(
            surge_m=("obs", long["value"].to_numpy()),
        ),
        coords=dict(
            figure=("obs", long["figure"].astype(str).to_numpy()),
            scope=("obs", long["scope"].astype(str).to_numpy()),
            region=("obs", long["region"].astype(str).to_numpy()),
            kind=("obs", long["kind"].fillna("unclassified").astype(str).to_numpy()),
            source_name=("obs", long["source_name"].astype(str).to_numpy()),
        ),
        attrs=dict(
            title="Storm surge maxima for Figures 5 (points) & 6 (regional)",
            summary="Values extracted from points/regional pickles using deterministic rules.",
            institution="Catherine R Jeffries / Virginia Tech, Geosciences",
            source=f"Derived from project pickles under {BASE}",
            history="Created by export_archive.py",
            conventions="CF-1.10 (discrete sampling style)",
            license="CC BY 4.0",
        ),
    )
    ds["surge_m"].attrs.update(dict(
        long_name="storm surge maximum",
        units="m"
    ))
    ds["figure"].attrs.update(dict(long_name="paper figure identifier"))
    ds["scope"].attrs.update(dict(long_name="aggregation scope", comment="point=fgmax_points, region=fg_max"))
    ds["region"].attrs.update(dict(long_name="region label"))
    ds["kind"].attrs.update(dict(long_name="plot bucket / data family"))
    ds["source_name"].attrs.update(dict(long_name="original column name (provenance)"))

    ds.to_netcdf(OUT_NC, encoding={"surge_m": {"zlib": True, "complevel": 3, "_FillValue": np.nan}})

    # Sanity print
    print("\nSanity checks:")
    print(long["figure"].value_counts(dropna=False))
    print(long.groupby(["figure","region","kind"]).size().reset_index(name="n"))
    print(f"\nWrote: {OUT_CSV} ({len(long)} rows)")
    print(f"Wrote: {OUT_NC}")
    print(f"Wrote convenience CSVs in: {OUT_CSV_DIR.resolve()}")

if __name__ == "__main__":
    main()
