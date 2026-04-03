#!/usr/bin/env python
"""Merge functional-connectivity firing-rate data with cell-type labels.

Reads ``functional_connectivity_2Hz.csv`` (FC session data) and
``OSI_DSI_neuropixels_v3.csv`` (v3 table with cell-type assignments).
Produces ``functional_connectivity_2Hz_with_cell_types.csv`` which contains
all FC neurons with their cell-type labels (where available) and additional
derived boolean columns for downstream convenience.
"""

import pandas as pd
import numpy as np

# --- Load inputs ---
fc_df = pd.read_csv("functional_connectivity_2Hz.csv", index_col=0)
print(f"Loaded {len(fc_df)} neurons from functional connectivity dataset")

cell_type_df = pd.read_csv("OSI_DSI_neuropixels_v3.csv", sep=" ")
print(f"Loaded {len(cell_type_df)} neurons with cell type information")

print("\nFunctional connectivity dataset columns:")
print(fc_df.columns.tolist())
print("\nCell type dataset columns:")
print(cell_type_df.columns.tolist())

print("\nFirst 3 rows of functional connectivity dataset (node_id):")
print(fc_df['node_id'].head(3))
print("\nFirst 3 rows of cell type dataset (ecephys_unit_id):")
print(cell_type_df['ecephys_unit_id'].head(3))

# --- Merge on unit ID ---
merged_df = pd.merge(
    fc_df,
    cell_type_df[['ecephys_unit_id', 'cell_type', 'Spont_Rate(Hz)']],
    left_on='node_id',
    right_on='ecephys_unit_id',
    how='left'
)

if 'ecephys_unit_id' in merged_df.columns:
    merged_df = merged_df.drop(columns=['ecephys_unit_id'])

# --- Derived columns ---
has_cell_type = merged_df['cell_type'].notna()
print(f"\nNeurons with cell type information: {has_cell_type.sum()} out of "
      f"{len(merged_df)} ({has_cell_type.sum()/len(merged_df)*100:.2f}%)")

merged_df['is_sst'] = merged_df['cell_type'].str.contains('SST', na=False)
merged_df['is_l6_sst'] = merged_df['cell_type'] == 'L6 SST'
merged_df['is_pv'] = merged_df['cell_type'].str.contains('PV', na=False)
merged_df['is_exc'] = merged_df['cell_type'].str.contains('Exc', na=False)
merged_df['cortical_layer'] = merged_df['cell_type'].str.extract(
    r'(L\d(?:/\d)?)', expand=False
)

cell_types = merged_df['cell_type'].value_counts()
print("\nCell type counts:")
print(cell_types)

l6_sst_count = merged_df['is_l6_sst'].sum()
print(f"\nFound {l6_sst_count} L6 SST cells in the dataset")

if l6_sst_count > 0:
    l6_sst_df = merged_df[merged_df['is_l6_sst']]
    print("\nDistribution of L6 SST cells by structure:")
    print(l6_sst_df['structure'].value_counts())

    responsive_l6_sst = l6_sst_df[l6_sst_df['is_responsive']]
    print(f"\nResponsive L6 SST cells: {len(responsive_l6_sst)} out of "
          f"{len(l6_sst_df)} ({len(responsive_l6_sst)/len(l6_sst_df)*100:.2f}%)")

    for ct in ['L6 SST', 'L6 PV', 'L6 Exc']:
        ct_df = merged_df[merged_df['cell_type'] == ct]
        if len(ct_df) > 0:
            responsive = ct_df[ct_df['is_responsive']]
            print(f"{ct}: {len(responsive)}/{len(ct_df)} responsive "
                  f"({len(responsive)/len(ct_df)*100:.2f}%)")

    print("\nDetailed information about L6 SST cells:")
    print(l6_sst_df[['node_id', 'structure', 'preferred_orientation',
                      'max_mean_rate(Hz)', 'Ave_Rate(Hz)',
                      'response_significance', 'is_responsive']].head(10))

    l6_sst_df.to_csv("l6_sst_cells.csv")
    print("Saved L6 SST data to l6_sst_cells.csv")

if has_cell_type.sum() > 0:
    typed_cells = merged_df[has_cell_type]
    print("\nDistribution of cells with type information by structure:")
    print(typed_cells['structure'].value_counts())

    print("\nCell type counts by structure:")
    structure_cell_type = pd.crosstab(
        typed_cells['structure'], typed_cells['cell_type']
    )
    print(structure_cell_type)

    if 'VISp' in typed_cells['structure'].values:
        print("\nVISp cell types:")
        visp_cells = typed_cells[typed_cells['structure'] == 'VISp']
        print(visp_cells['cell_type'].value_counts())

# --- Save ---
merged_df.to_csv("functional_connectivity_2Hz_with_cell_types.csv")
print(f"\nSaved dataset with cell type information to "
      f"functional_connectivity_2Hz_with_cell_types.csv")

if l6_sst_count < 5:
    print("\nNote: Found very few L6 SST cells in the functional "
          "connectivity dataset.")
    sst_in_full = cell_type_df['cell_type'].str.contains(
        'SST', na=False
    ).sum()
    l6_sst_in_full = (cell_type_df['cell_type'] == 'L6 SST').sum()
    print(f"Total SST cells in OSI_DSI_neuropixels_v3.csv: {sst_in_full}")
    print(f"Total L6 SST cells in OSI_DSI_neuropixels_v3.csv: "
          f"{l6_sst_in_full}")
