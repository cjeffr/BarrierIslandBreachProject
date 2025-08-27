#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot 3-panel histograms (West/Central/East) from the archived data produced by export_archive.py.
Works with either 'archive_long.csv' or 'archive.nc'.

Usage examples:
  python plot_from_archive.py
  # or from a notebook:
  # plot_fig("archive_long.csv", figure="Fig4", style="./mystyle.mplstyle")
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: xarray is only needed if you pass a .nc file
try:
    import xarray as xr
except Exception:
    xr = None

# -------------------------
# Config (colors & labels)
# -------------------------
PALETTE = {
    "width_depth": "#D55E00",  # orange-brown
    "width":       "#009E73",  # green
    "depth":       "#56B4E9",  # blue
    "locations":   "#CC79A7",  # magenta
    "east":        "#E69F00",  # orange
    "west":        "grey",     # grey
}

ORDER = ["width_depth", "width", "depth", "locations", "east", "west"]
DISPLAY = {
    "width_depth": "Width and Depth",
    "width":       "Width",
    "depth":       "Depth",
    "locations":   "Locations",
    "east":        "East",
    "west":        "West",
}

REGIONS = ["west", "central", "east"]  # display order

# --------------------------------------
# Loaders (CSV or NetCDF, long/“tidy”)
# --------------------------------------
def load_archive(path: str | Path) -> pd.DataFrame:
    """
    Returns a long DataFrame with columns:
    ['figure','scope','region','kind','source_name','value']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".nc":
        if xr is None:
            raise ImportError("xarray not available; install it or use the CSV.")
        ds = xr.load_dataset(path)
        # Build a long dataframe from per-observation coords
        df = pd.DataFrame({
            "figure":      ds["figure"].values.astype(str),
            "scope":       ds["scope"].values.astype(str),
            "region":      ds["region"].values.astype(str),
            "kind":        ds["kind"].values.astype(str),
            "source_name": ds["source_name"].values.astype(str),
            "value":       ds["surge_m"].values,
        })
    else:
        raise ValueError("Pass a '.csv' or '.nc' file")

    # Basic hygiene
    df = df.dropna(subset=["value"]).copy()
    df["figure"] = df["figure"].astype(str)
    df["scope"] = df["scope"].astype(str)
    df["region"] = df["region"].astype(str)
    df["kind"] = df["kind"].astype(str)
    return df


# --------------------------------------
# Plot routine
# --------------------------------------
def plot_fig(archive_path: str | Path,
             figure: str = "Fig4",
             style: str | None = None,
             savepath: str | None = None):
    """
    Plot a 1x3 panel (West/Central/East) for the requested figure label
    (e.g., "Fig4" for point maxima, "Fig5" for regional maxima).
    """
    from matplotlib.patches import Patch

    if style:
        plt.style.use(style)

    df = load_archive(archive_path)
    df = df[df["figure"] == figure].copy()
    if df.empty:
        raise ValueError(f"No rows for figure={figure!r} in {archive_path}")

    # Compute a common bin edge array across all data (like your combined bins)
    all_vals = df["value"].to_numpy()
    bins = np.histogram(all_vals, bins=20)[1]

    # Prepare canvas
    plt.rcParams.update({'font.size': 16*1.95})  # match your scaling
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    names = ["West", "Central", "East"]

    for idx, region in enumerate(REGIONS):
        ax = axes[idx]
        ax.set_title(names[idx])

        # Plot each bucket (only if present for the region)
        handles = []
        labels = []

        for kind in ORDER:
            sub = df[(df["region"] == region) & (df["kind"] == kind)]
            if sub.empty:
                continue
            vals = sub["value"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            weights = np.ones_like(vals) / len(vals)

            # filled + outline (like your hist_nine)
            h1 = ax.hist(vals, bins=bins, weights=weights,
                         color=PALETTE[kind], histtype="stepfilled", alpha=0.2)
            h2 = ax.hist(vals, bins=bins, weights=weights,
                         color=PALETTE[kind], histtype="step", alpha=0.7)

            # Keep one handle per kind for legend
            # Keep one handle per kind for legend (use a filled proxy)
            if DISPLAY[kind] not in labels:
                proxy = Patch(
                    facecolor=PALETTE[kind],
                    edgecolor=PALETTE[kind],
                    alpha=0.2,           # matches the filled hist
                    label=DISPLAY[kind],
                )
                handles.append(proxy)
                labels.append(DISPLAY[kind])
            


        # Axes formatting to match your original
        ax.set_xlim(0.55, 3.55)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.75, 1.5, 2.25, 3.0])
        ax.set_xticklabels(['0.75', '1.5', '2.25', '3.0'])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

        # Only collect legend handles from the first axis to place globally later
        if idx == 0:
            first_handles, first_labels = handles, labels

    # Global legend (below plots)
    fig.legend(first_handles, first_labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.20), ncol=6, fontsize=14/0.95, frameon=False)

    # Global labels (match your fig.text usage)
    fig.text(0.5, 0.0001, 'Storm Surge (m)', ha='center', va='center', fontsize=18)
    fig.text(0.06, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=18)
    fig.subplots_adjust(bottom=0.25, left=0.12, wspace=0.25)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig, axes


# -------------------------
# CLI entrypoint (optional)
# -------------------------
if __name__ == "__main__":
    # Adjust the path/figure/style as needed:
    #   - "archive_long.csv"  or  "archive.nc"
    P = "/home/catherinej/data_archive/histogram_archive_long.csv" if Path("archive_long.csv").exists() else "/home/catherinej/data_archive/histogram_archive.nc"
    print(f"Loading: {P}")
    fig, axes = plot_fig(P, figure="Fig5", style="/home/catherinej/BarrierIslandBreachProject/notebooks/mystyle.mplstyle", savepath="fig_check_fig5.pdf")
    plt.show()

    # Quick second plot for Fig5:
    fig, axes = plot_fig(P, figure="Fig6", style="/home/catherinej/BarrierIslandBreachProject/notebooks/mystyle.mplstyle", savepath="fig_check_fig6.pdf")
    plt.show()
