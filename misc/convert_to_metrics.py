#!/usr/bin/env python
"""Build an OSI/DSI v3 CSV for a target visual area.

Reads cortical_metrics_1.4.csv (or a compatible metrics file), applies
quality-control filters, converts compact cell-type codes to human-readable
labels (e.g. ``EXC_L23`` → ``L2/3 Exc``), and writes a space-delimited CSV
containing DSI, OSI, preferred angle, firing rates and cell type.

SST and VIP cells are kept from *all* visual cortical areas (because these
populations are small), while Exc and PV cells are restricted to the
requested ``--area``.
"""

import argparse
import pandas as pd
import numpy as np


def convert_layer(layer_string):
    """Convert compact cell-type code to human-readable label."""
    if pd.isna(layer_string):
        return np.nan
    typedic = {"E": "Exc", "P": "PV", "S": "SST", "V": "VIP", "A": "Htr3a"}
    layerdic = {"3": "2/3", "4": "4", "5": "5", "6": "6", "1": "1"}
    return "L" + layerdic[layer_string[-1]] + " " + typedic[layer_string[0]]


def main():
    parser = argparse.ArgumentParser(
        description="Build OSI/DSI v3 CSV for a target area with SST/VIP "
                    "from all cortex and Exc/PV restricted to area."
    )
    parser.add_argument(
        "--area", default="VISp",
        help="Target area acronym to restrict Exc/PV "
             "(e.g., VISp for V1, VISl for LM)",
    )
    parser.add_argument(
        "--metrics", default="cortical_metrics_1.4.csv",
        help="Input cortical metrics CSV",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output filename. Defaults to OSI_DSI_neuropixels_{AREA}_v3.csv",
    )
    args = parser.parse_args()

    target_area = args.area
    out_name = args.output or f"OSI_DSI_neuropixels_{target_area}_v3.csv"

    # --- Load metrics table ---
    df_all = pd.read_csv(args.metrics, index_col=0, low_memory=False)
    print(f"Loaded {len(df_all)} rows from {args.metrics}")

    # --- Selection rule ---
    # Keep SST and VIP across cortex (any VIS*) since these are small
    # populations.  Restrict Exc and PV to the requested area only.
    exclude = df_all["cell_type"].isna() | (
        (
            df_all["cell_type"].str.contains("EXC")
            | df_all["cell_type"].str.contains("PV")
        )
        & (df_all["ecephys_structure_acronym"] != target_area)
    )
    df = df_all[~exclude].copy()

    # --- Quality control ---
    df["qc_pass"] = (
        (df["presence_ratio"] > 0.9)
        & (df["amplitude_cutoff"] < 0.1)
        & (df["isi_violations"] < 0.5)
    )
    df = df[df["qc_pass"] == True].copy()
    print(f"After QC: {len(df)} cells")

    # --- Convert cell-type codes ---
    df["cell_type"] = df["cell_type"].apply(convert_layer)

    # --- Select output columns ---
    pickup_metrics = [
        "DSI",
        "OSI",
        "preferred_angle",
        "max_mean_rate(Hz)",
        "Ave_Rate(Hz)",
        "cell_type",
        "firing_rate_sp",
    ]
    simpledf = df[pickup_metrics].rename(
        columns={"firing_rate_sp": "Spont_Rate(Hz)"}
    )

    # --- Save ---
    simpledf.to_csv(out_name, sep=" ")
    print(f"Wrote {len(simpledf)} rows to {out_name}")


if __name__ == "__main__":
    main()
