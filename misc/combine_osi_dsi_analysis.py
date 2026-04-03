#!/usr/bin/env python
"""Combine an OSI/DSI v3 table with functional-connectivity firing rates.

For each v3 cell that also appears in the FC dataset the firing-rate columns
(``max_mean_rate(Hz)``, ``Ave_Rate(Hz)``, ``Spont_Rate(Hz)``) are updated
with the FC values.  FC-only cells that are *not* in v3 but belong to the
target ``--area`` are appended with NaN for OSI/DSI columns.  Cell types for
newly appended cells are back-filled from ``cortical_metrics_1.4.csv`` when
available.  Finally, ``firing_rate_ns`` (natural-scenes firing rate) is
attached from the same cortical-metrics file.

Rows without a valid ``cell_type`` are dropped from the output.
"""

import argparse
import csv

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Combine OSI/DSI v3 with FC rates for a target area"
    )
    parser.add_argument(
        "--v3", required=True,
        help="Path to OSI_DSI_neuropixels_{AREA}_v3.csv",
    )
    parser.add_argument(
        "--fc", default="functional_connectivity_2Hz_with_cell_types.csv",
        help="FC CSV with rates",
    )
    parser.add_argument("--out", required=True, help="Output v4 CSV path")
    parser.add_argument(
        "--area", required=False,
        help="Target area acronym (e.g., VISp, VISl). If provided, only "
             "FC-only new cells matching this area are appended; existing "
             "v3 cells are always updated.",
    )
    args = parser.parse_args()

    osi_v3_file = args.v3
    fc_file = args.fc
    output_v4_file = args.out

    print("--- Starting Data Merge and Update Process ---")

    # --- Load OSI v3 Data ---
    print(f"Loading OSI v3 data from: {osi_v3_file}")
    osi_v3_df = pd.read_csv(
        osi_v3_file, sep=" ", index_col="ecephys_unit_id", quotechar='"'
    )
    osi_v3_df.columns = [col.strip() for col in osi_v3_df.columns]
    print(f"Loaded OSI v3. Shape: {osi_v3_df.shape}. "
          f"Columns: {list(osi_v3_df.columns)}")

    rate_cols_v3 = [
        "max_mean_rate(Hz)", "Ave_Rate(Hz)", "Spont_Rate(Hz)",
        "DSI", "OSI", "preferred_angle",
    ]
    for col in rate_cols_v3:
        if col in osi_v3_df.columns:
            osi_v3_df[col] = pd.to_numeric(osi_v3_df[col], errors="coerce")

    # --- Load FC Data ---
    print(f"Loading FC data from: {fc_file}")
    fc_raw_df = pd.read_csv(fc_file)
    if "node_id" not in fc_raw_df.columns:
        raise KeyError("Column 'node_id' not found in FC file!")
    fc_raw_df.rename(columns={"node_id": "ecephys_unit_id"}, inplace=True)
    fc_raw_df.set_index("ecephys_unit_id", inplace=True)
    print(f"FC data shape after indexing: {fc_raw_df.shape}")

    # --- Prepare FC rate columns ---
    rate_cols = [
        c for c in ["max_mean_rate(Hz)", "Ave_Rate(Hz)", "Spont_Rate(Hz)"]
        if c in fc_raw_df.columns
    ]
    if len(rate_cols) == 0:
        raise KeyError("FC file missing expected rate columns")
    fc_rates_df = fc_raw_df[rate_cols].copy()
    print(f"FC rate columns: {list(fc_rates_df.columns)}")

    # --- Update v3 Data with FC Rates ---
    print("Updating OSI v3 data with FC rates for matching cells...")
    osi_v3_df.update(fc_rates_df, overwrite=True)
    print("Update complete.")

    # --- Identify and Prepare New Cells from FC ---
    new_cell_indices = fc_rates_df.index.difference(osi_v3_df.index)
    print(f"Found {len(new_cell_indices)} new cells present only in FC data.")

    if not new_cell_indices.empty:
        if args.area:
            area_col = (
                "structure" if "structure" in fc_raw_df.columns
                else (
                    "ecephys_structure_acronym"
                    if "ecephys_structure_acronym" in fc_raw_df.columns
                    else None
                )
            )
            if area_col is None:
                print("Warning: no area column in FC data; "
                      "skipping FC-only cells.")
                new_filtered_indices = []
            else:
                new_filtered_indices = [
                    i for i in new_cell_indices
                    if str(fc_raw_df.loc[i, area_col]) == args.area
                ]
        else:
            new_filtered_indices = list(new_cell_indices)

        new_fc_cells_df = fc_rates_df.loc[new_filtered_indices].copy()
        print(f"New FC cells for area: {new_fc_cells_df.shape}")

        # Add missing v3 columns initialised as NaN
        missing_v3_cols = [
            col for col in osi_v3_df.columns
            if col not in new_fc_cells_df.columns
        ]
        for col in missing_v3_cols:
            new_fc_cells_df[col] = np.nan

        # Back-fill cell_type from cortical metrics when possible
        if "cell_type" in missing_v3_cols and len(new_filtered_indices) > 0:
            try:
                metrics_df = pd.read_csv("cortical_metrics_1.4.csv")
                metrics_df["ecephys_unit_id"] = (
                    metrics_df["ecephys_unit_id"].astype(int)
                )
                metrics_df.set_index("ecephys_unit_id", inplace=True)

                def _convert_layer(layer_string):
                    if pd.isna(layer_string):
                        return np.nan
                    typedic = {
                        "E": "Exc", "P": "PV", "S": "SST",
                        "V": "VIP", "A": "Htr3a",
                    }
                    layerdic = {
                        "3": "2/3", "4": "4", "5": "5", "6": "6", "1": "1",
                    }
                    s = str(layer_string)
                    return ("L" + layerdic.get(s[-1], s[-1]) + " "
                            + typedic.get(s[0], s[0]))

                ct = metrics_df.loc[
                    new_fc_cells_df.index.intersection(metrics_df.index),
                    "cell_type",
                ].apply(_convert_layer)
                new_fc_cells_df.loc[ct.index, "cell_type"] = ct
            except Exception as e:
                print(f"Warning: could not backfill cell_type: {e}")

        new_fc_cells_df = new_fc_cells_df[osi_v3_df.columns]

        print("Concatenating updated v3 data with new FC cell data...")
        final_v4_df = pd.concat([osi_v3_df, new_fc_cells_df])
        print(f"Final shape: {final_v4_df.shape}")
    else:
        print("No new cells from FC. Final dataframe is the updated v3.")
        final_v4_df = osi_v3_df

    final_v4_df.sort_index(inplace=True)

    # --- Attach natural-scenes firing rate from cortical metrics ---
    try:
        metrics_df_all = pd.read_csv("cortical_metrics_1.4.csv")
        metrics_df_all["ecephys_unit_id"] = (
            metrics_df_all["ecephys_unit_id"].astype(int)
        )
        metrics_df_all.set_index("ecephys_unit_id", inplace=True)
        if "firing_rate_ns" in metrics_df_all.columns:
            final_v4_df["firing_rate_ns"] = metrics_df_all[
                "firing_rate_ns"
            ].reindex(final_v4_df.index)
            print("Attached firing_rate_ns from cortical_metrics_1.4.csv")
        else:
            print("Warning: 'firing_rate_ns' not found in cortical metrics")
    except Exception as e:
        print(f"Warning: could not attach firing_rate_ns: {e}")

    # --- Clean and Filter Cell Types ---
    final_v4_df["cell_type"] = (
        final_v4_df["cell_type"].astype(str).str.strip('"')
    )
    final_v4_df_filtered = final_v4_df[
        final_v4_df["cell_type"] != "nan"
    ].copy()
    print(f"After cell_type filter: {final_v4_df_filtered.shape}")

    # --- Save ---
    print(f"Saving to: {output_v4_file}")
    final_v4_df_filtered.to_csv(
        output_v4_file,
        sep=" ",
        index=True,
        index_label="ecephys_unit_id",
        na_rep="NaN",
        header=True,
        quoting=csv.QUOTE_MINIMAL,
    )
    print("--- Data Merge and Update Process Finished Successfully ---")


if __name__ == "__main__":
    main()
