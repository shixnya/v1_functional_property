#!/usr/bin/env python
"""Extract firing-rate statistics from Functional Connectivity sessions.

Processes drifting-grating responses (2 Hz temporal frequency, 0.8 contrast,
75 repeats) from all FC sessions in the Allen Institute Neuropixels Visual
Coding dataset.  For each visual-cortex neuron the script computes:

- preferred orientation (from 4 directions)
- maximum and average mean firing rates
- response significance (quasi-Poisson test)
- per-orientation angular response vector

Because FC sessions present only 4 grating directions (not 8), only firing-
rate statistics are extracted — direction/orientation selectivity indices
(DSI/OSI) are *not* computed here.

Requirements
------------
- allensdk
- numpy, pandas, tqdm
- lazyscience  (local module in this directory, provides quasi_poisson_sig_test_counts)

Usage
-----
    python misc/functional_connectivity_2Hz.py --manifest /path/to/manifest.json

The script caches per-session results as HDF5 files under ``fc_h5s/`` so that
re-runs skip already-processed sessions.  Final output is written to
``functional_connectivity_2Hz.csv`` (and ``.h5``).
"""

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# lazyscience lives in the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lazyscience as lz

from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
    EcephysProjectCache,
)

# ---------------------------------------------------------------------------
# These module-level variables are populated in main() and read by the
# multiprocessing worker (process_session_id).
# ---------------------------------------------------------------------------
cache = None
fc_sessions = None


def calculate_firing_rates_df(cache, session_id):
    """Compute firing-rate metrics for one FC session.

    Returns a DataFrame with one row per visual-cortex unit.
    Per-session results are cached to ``fc_h5s/{session_id}.h5``.
    """
    path = f"fc_h5s/{session_id}.h5"
    if os.path.exists(path):
        return pd.read_hdf(path)

    session = cache.get_session_data(session_id)

    # Get drifting-grating stimulus table
    try:
        dg_table = session.get_stimulus_table(["drifting_gratings_75_repeats"])
    except Exception:
        print(f"No drifting_gratings_75_repeats in session {session_id}")
        return pd.DataFrame()

    # Filter for 2 Hz temporal frequency and 0.8 contrast
    dg_table_filtered = dg_table[
        (dg_table["temporal_frequency"] == 2.0)
        & (dg_table["contrast"] == 0.8)
    ].sort_values(by="orientation")

    if len(dg_table_filtered) == 0:
        print(f"No 2 Hz / 0.8 contrast gratings in session {session_id}")
        return pd.DataFrame()

    orientations = dg_table_filtered.orientation.unique()
    n_orientations = len(orientations)

    # Visual-cortex units only
    units = session.units
    vcunit_id = units.index[
        units.ecephys_structure_acronym.str.contains("VIS")
    ]
    if len(vcunit_id) == 0:
        print(f"No visual cortex units in session {session_id}")
        return pd.DataFrame()

    # Spike counts: stimulus window (0–2 s) and spontaneous window (2–4 s)
    vc_dg = session.presentationwise_spike_counts(
        [0.0, 2.0], dg_table_filtered.index, vcunit_id
    )
    vc_spont = session.presentationwise_spike_counts(
        [2.0, 4.0], dg_table_filtered.index, vcunit_id
    )

    n_cells = len(vcunit_id)
    rep_counts = dg_table_filtered.orientation.value_counts()
    max_reps = rep_counts.max()

    ori_to_idx = {ori: i for i, ori in enumerate(sorted(orientations))}

    spike_array = np.full((n_orientations, max_reps, n_cells), np.nan)
    spont_array = np.full((n_orientations, max_reps, n_cells), np.nan)

    for _, row in dg_table_filtered.iterrows():
        ori_idx = ori_to_idx[row.orientation]
        rep_count = np.sum(
            (dg_table_filtered.orientation == row.orientation)
            & (dg_table_filtered.index <= row.name)
        )
        if rep_count <= max_reps:
            rep_idx = rep_count - 1
            spikes = np.array(
                vc_dg.sel(stimulus_presentation_id=row.name)
            )
            spontaneous = np.array(
                vc_spont.sel(stimulus_presentation_id=row.name)
            )
            spike_array[ori_idx, rep_idx, :] = spikes
            spont_array[ori_idx, rep_idx, :] = spontaneous

    # Mean firing rates (Hz) — divide counts by 2 s window
    spike_rates = np.nanmean(spike_array, axis=1).T / 2.0
    # spont_rates = np.nanmean(spont_array, axis=1).T / 2.0  # not stored

    # Response significance (quasi-Poisson test per cell)
    resp_sig = []
    for cell_idx in range(n_cells):
        cell_spikes = spike_array[:, :, cell_idx]
        cell_spont = spont_array[:, :, cell_idx]
        sig = lz.quasi_poisson_sig_test_counts(
            response_counts=cell_spikes,
            window_size1=2.0,
            spont_counts=cell_spont,
            window_size2=2.0,
            unit_axis=0,
        )
        resp_sig.append(np.min(sig))

    preferred_ori_idx = np.argmax(spike_rates, axis=1)
    preferred_orientations = np.array(sorted(orientations))[preferred_ori_idx]
    max_rates = np.nanmax(spike_rates, axis=1)
    avg_rates = np.nanmean(spike_rates, axis=1)

    angular_response = [spike_rates[i, :] for i in range(spike_rates.shape[0])]

    structures = [
        units.loc[uid, "ecephys_structure_acronym"] for uid in vcunit_id
    ]

    result_df = pd.DataFrame(
        {
            "node_id": vcunit_id,
            "structure": structures,
            "preferred_orientation": preferred_orientations,
            "max_mean_rate(Hz)": max_rates,
            "Ave_Rate(Hz)": avg_rates,
            "response_significance": resp_sig,
            "is_responsive": np.array(resp_sig) < 0.01,
            "angular_response": angular_response,
        }
    )

    os.makedirs("fc_h5s", exist_ok=True)
    result_df.to_hdf(f"fc_h5s/{session_id}.h5", key="df", mode="w")
    return result_df


def process_session_id(session_id):
    """Multiprocessing worker — uses module-level ``cache``."""
    return calculate_firing_rates_df(cache, session_id)


def main():
    global cache, fc_sessions

    parser = argparse.ArgumentParser(
        description="Extract FC firing-rate statistics from the "
                    "Allen Neuropixels data cache."
    )
    parser.add_argument(
        "--manifest",
        default="/local1/data/ecephys_cache_dir/manifest.json",
        help="Path to the EcephysProjectCache manifest.json",
    )
    args = parser.parse_args()

    cache = EcephysProjectCache.from_warehouse(manifest=args.manifest)
    sessions = cache.get_session_table()
    fc_sessions = sessions[sessions.session_type == "functional_connectivity"]
    print(f"# Functional Connectivity sessions: {len(fc_sessions)}")

    # --- Process all sessions in parallel ----------------------------------
    os.makedirs("fc_h5s", exist_ok=True)
    pool = mp.Pool(mp.cpu_count())
    all_results = list(
        tqdm(
            pool.imap(process_session_id, fc_sessions.index.values),
            total=len(fc_sessions),
        )
    )
    pool.close()

    df_all = pd.concat([df for df in all_results if not df.empty])

    print(f"Saving results for {len(df_all)} units")
    df_all.to_csv("functional_connectivity_2Hz.csv")
    df_all.to_hdf("functional_connectivity_2Hz.h5", "df", mode="w")

    print(f"Processed {len(all_results)} sessions")
    print(f"Found {len(df_all)} total units")
    print(f"Responsive units: {df_all['is_responsive'].sum()}")
    print("\nUnits by structure:")
    print(df_all["structure"].value_counts())


if __name__ == "__main__":
    main()
