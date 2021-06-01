# %%  import the table

import pandas as pd
import numpy as np

df_all = pd.read_csv("cortical_metrics.csv")

# Exc and PV have sufficient number of cells, so we'll filter out non-V1 Exc and PV.
exclude = (
    df_all["cell_type"].isnull()
    | df_all["cell_type"].str.contains("EXC")
    | df_all["cell_type"].str.contains("PV")
) & (df_all["ecephys_structure_acronym"] != "VISp")

df = df_all[~exclude]
print(f"Original: {df_all.shape[0]} cells,   filtered: {df.shape[0]} cells")


# %% count the number of cells in each session

bo_counts = df[df["session_type"] == "brain_observatory_1.1"][
    "cell_type"
].value_counts()
fc_counts = df[df["session_type"] == "functional_connectivity"][
    "cell_type"
].value_counts()

# Some cells have very large values of RF. They are likely not-good fits, so ignore.
df.loc[(df["width_rf"] > 100), "width_rf"] = np.nan
df.loc[(df["height_rf"] > 100), "height_rf"] = np.nan

# define contrast reponse types
types = ["high", "band", "low"]
type_fracs = {}
type_fracs_err = {}
contrast_responsive = df["sig0.01"] == True
contrast_responsive_counts = df[contrast_responsive]["cell_type"].value_counts()
for t in types:
    type_counts = df[(df["best_model"] == t) & contrast_responsive][
        "cell_type"
    ].value_counts()
    n = contrast_responsive_counts
    p = type_counts / n
    type_fracs[t] = p
    type_fracs_err[t] = np.sqrt(p * (1 - p) / n)


# set group statistics, and insert the cell counts for convenience
resource_tables = {}
resource_tables["median"] = df.groupby("cell_type").median()
resource_tables["mean"] = df.groupby("cell_type").mean()
resource_tables["sem"] = df.groupby("cell_type").sem()

for t in resource_tables.values():
    # insert counts and fractional elements that is common for all 3 tables
    t.insert(0, "n_cells_BO", bo_counts)
    t.insert(1, "n_cells_FC", fc_counts)
    # t.insert(-1, "frac_highpass", type_fracs["high"])
    t["frac_highpass"] = type_fracs["high"]
    t["frac_bandpass"] = type_fracs["band"]
    t["frac_lowpass"] = type_fracs["low"]

# sem is a special case. re-insert the estimated error of the fraction
resource_tables["sem"]["frac_highpass"] = type_fracs_err["high"]
resource_tables["sem"]["frac_bandpass"] = type_fracs_err["band"]
resource_tables["sem"]["frac_lowpass"] = type_fracs_err["low"]

# %% pick up elements that we'd like to show, and display as a table
selected_metrics = [
    "n_cells_BO",
    "n_cells_FC",
    "g_osi_dg",
    "g_dsi_dg",
    "pref_sf_sg",
    "pref_tf_dg",
    "firing_rate_rf",
    "firing_rate_dg",
    "width_rf",
    "height_rf",
    "fano_dg",
    "frac_highpass",
    "frac_bandpass",
    "frac_lowpass",
]

# For the full list of metrics, refer to list(df.columns)
#
# See https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
# for detailed descriptions of the metrics.

# %%
pd.set_option("display.float_format", "{:0.2f}".format)
print("Median functional responses")
resource_tables["median"][selected_metrics]

# %%
print("Mean functional responses")
resource_tables["mean"][selected_metrics]

# %%
print("SEM of functional responses")
resource_tables["sem"][selected_metrics]
