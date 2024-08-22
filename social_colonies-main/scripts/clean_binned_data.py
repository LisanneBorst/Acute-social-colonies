"""
Main pre-processing script to clean binned data from SocialScan

TODO: doc
"""

import pandas as pd
import numpy as np
import os
import warnings
from utils import *

# Define These TODO: Settings file
data_folder_path = "binned_data"
output_folder_path = "processed_data"
experiment_name = "P8-acute-col"
include_video_name = True

# Behavior Regex patterns (Add more if you want to export more)
behavior_and_regex = {
    "social_contact": r"Social Contact.*\[? ?[^#^\d]{}(?:\[\#\d+\])?",
    "social_approach": r"Social Approach \[? ?{}(?:\[\#\d+\])? to \d+(?:\[\#\d+\])? ?\]?",
    "social_passive_approach": r"Social Approach \[? ?\d+(?:\[\#\d+\])? to {}(?:\[\#\d+\])? ?\]?",
    "social_follow": r"Social Follow \[? ?{}(?:\[\#\d+\])? follow \d+(?:\[\#\d+\])? ?\]?",
    "social_leave": r"Social Leave \[? ?{}(?:\[\#\d+\])? from \d+(?:\[\#\d+\])? ?\]?",
    "social_passive_follow": r"Social Follow \[? ?\d+(?:\[\#\d+\])? follow {}(?:\[\#\d+\])? ?\]?",
    "social_sniff": r"Social Sniff \[? ?{}(?:\[\#\d+\])? sniff \d+(?:\[\#\d+\])? ?\]?",
    "social_passive_sniff": r"Social Sniff \[? ?\d(?:\[\#\d+\])? sniff {}(?:\[\#\d+\])? ?\]?",
    "hide_in_nest": r"Animal {}.* Hide in Nest",
    "contact_nest": r"Animal {}.* Contact Nest",
    "social_hide": r"Social Hide.*\[? ?[^#^\d]{}(?:\[\#\d+\])?.+ (?=Nest \d+)",
    "in_area": r"Area:Mouse {}.*In",
}

# Main Looperson

# Hold all dataframes
all_outs = []

for file in os.listdir(data_folder_path):
    if not file.endswith(".xlsx"):
        continue
    print(f"Analyzing {file}...")
    batch, colony = get_batch_colony(file)
    print(f"Selected batch: {batch}, selected colony {colony}")

    # Read data
    beh = pd.read_excel(
        os.path.join(data_folder_path, file), sheet_name="All Evt Detail"
    )
    bin = pd.read_excel(os.path.join(data_folder_path, file), sheet_name="Bin Measure")

    # ============ Clean Event Measures
    beh.columns = beh.iloc[5].reset_index(drop=True)
    beh = beh.rename(
        {
            "Events": "Measure",
            "Total Bouts": "Bouts",
            "Total Duration(Second)": "Duration",
        },
        axis=1,
    )
    beh = (
        beh.drop(np.arange(6))
        .loc[:, ["Trial ID", "Video Name", "Measure", "Bouts", "Duration"]]
        .dropna()
        .reset_index(drop=True)
    )
    beh = process_measures(beh)

    # =========== Clean Bin Measures
    bin.columns = bin.iloc[5].reset_index(drop=True)
    bin = (
        bin.drop(np.arange(6))
        .loc[:, ["Trial ID", "Video Name", "Measure", "Bin1"]]
        .dropna()
        .reset_index(drop=True)
    )
    bin = process_measures(bin)

    # =========== Parse Event Measures
    # Identify scored behaviors
    scored_behaviors = get_exported_behaviors(beh["Measure"].unique())
    print(f"Scored behaviors: {scored_behaviors}")

    # Find max animals from bin measures
    max_animals = find_max_animals(bin)
    print(f"Max animals: {max_animals}")

    out = pd.DataFrame()
    # Loop over all behaviors
    for behavior in scored_behaviors:
        # Loop animals
        for a in range(1, max_animals + 1):
            # Format behavior-regex
            pattern = behavior_and_regex[behavior].format(a)
            print(f"Pattern for behavior {behavior}, animal {a}: {pattern}")

            # Exract behavior-specific data
            df = beh[beh["Measure"].str.contains(pattern)]
            print(f"Behavior-specific data for {behavior}, animal {a}: {df}")

            if df.empty:
                print(f"No data found for behavior {behavior}, animal {a}")
                continue

            # Reshape factor
            res_fac = (
                beh[beh["Measure"].str.contains(pattern)]["Measure"].unique().shape[0]
            )

            for day in beh["day_count"].unique():

                # Extract Column info
                day_data = df[df["day_count"] == day]
                day_bin = day_data["day_bin"].unique()
                print(f"Day data for {behavior}, animal {a}, day {day}: {day_data}")

                if day_data.empty:
                    print(f"No day data found for {behavior}, animal {a}, day {day}")
                    continue

                # Check that Video Name has len of 1
                if day_data["Video Name"].unique().shape[0] > 1:
                    warnings.warn(
                        f'More than one Video Name found for {behavior} on day {day}. Check the data {day_data["Video Name"].unique()}'
                    )

                all_outs.append(
                    pd.DataFrame(
                        {  # TODO: More columns
                            "batch": batch,
                            "arena": colony,
                            "day": day,
                            "day_bin": day_bin,
                            "session_bin": day_data["session_bin"].unique(),
                            "bouts": day_data["Bouts"]
                            .to_numpy()
                            .reshape(day_bin.max(), res_fac)
                            .sum(axis=1),
                            "duration": day_data["Duration"]
                            .to_numpy()
                            .reshape(day_bin.max(), res_fac)
                            .sum(axis=1),
                            "animal_in_event_record": a,
                            "Video Name": day_data["Video Name"].unique()[
                                0
                            ],  # Not sure if this is correct so TODO add a warning if the len > 1
                            "real_time": day_data["real_time"].unique()[0],
                            "real_date": day_data["real_date"].unique()[0],
                            "behavior": behavior,
                        }
                    )
                )
                print(f"Parsed event measures for {behavior}, animal {a}, day {day}")

    # =========== Parse Bin Measures
    for a in range(1, max_animals + 1):
        # distance moved
        _ = bin[bin["Measure"].str.contains(f"Mouse{a}-Center")]
        print(f"Distance moved data for animal {a}: {_.shape}")
        all_outs.append(
            pd.DataFrame(
                {
                    "batch": batch,
                    "arena": colony,
                    "day": _["day_count"],
                    "day_bin": _["day_bin"],
                    "session_bin": _["session_bin"],
                    "distance": _["Bin1"],
                    "animal_in_event_record": a,
                    "Video Name": _["Video Name"],
                    "real_time": _["real_time"],
                    "real_date": _["real_date"],
                    "behavior": "distance_moved",
                }
            )
        )

        # social distance
        _ = bin[bin["Measure"].str.contains(f"Mean Distance Between.+Mouse{a}")]
        print(f"Social distance data for animal {a}: {_.shape}")
        res_fac = _["Measure"].unique().shape[0]

        for day in _["day_count"].unique():

            day_data = _[_["day_count"] == day]
            day_bin = day_data["day_bin"].unique()
            print(f"Day data for social distance, animal {a}, day {day}: {day_data}")

            all_outs.append(
                pd.DataFrame(
                    {
                        "batch": batch,
                        "arena": colony,
                        "day": day,
                        "day_bin": day_bin,
                        "session_bin": day_data["session_bin"].unique(),
                        "distance": day_data["Bin1"]
                        .to_numpy()
                        .reshape(res_fac, day_bin.max())
                        .mean(axis=0),
                        "animal_in_event_record": a,
                        "Video Name": day_data["Video Name"].unique()[0],
                        "real_time": day_data["real_time"].unique()[0],
                        "real_date": day_data["real_date"].unique()[0],
                        "behavior": "social_distance",
                    }
                )
            )
            print(f"Parsed bin measures for social distance, animal {a}, day {day}")

if all_outs:
    out = pd.concat(all_outs, ignore_index=True)
    out.to_excel(
        f"{output_folder_path}/processed_data_{experiment_name}.xlsx", index=False
    )
    print("Data concatenated and saved to Excel.")
else:
    print("No data to concatenate.")

# Make a metadata file that will need to be filled in later
if include_video_name:
    metadata = out[
        ["batch", "arena", "animal_in_event_record", "Video Name"]
    ].drop_duplicates(ignore_index=True)
else:
    metadata = out[["batch", "arena", "animal_in_event_record"]].drop_duplicates(
        ignore_index=True
    )

# Add some "suggested" columns with info to be included
metadata["animal_id"] = None
# metadata["genotype"] = None
metadata["surgery"] = None
metadata["treatment"] = None

metadata.to_excel(
    f"{output_folder_path}/experiment_metadata_{experiment_name}.xlsx", index=False
)

print("Metadata saved to Excel.")

print("Done!")


"""
Extra code to unblind treatments A or B to Saline or CNO
"""

# Change treatment A/B to CNO/Saline
df = pd.read_excel("processed_data\experiment_metadata_P8-acute-col.xlsx")
df.rename(columns={"treatment": "treatment_AB"}, inplace=True)
# Insert a new empty 'treatment' column next to 'treatment_AB'
df.insert(df.columns.get_loc("treatment_AB") + 1, "treatment", "")

# Mapping conditions to treatment labels
treatment_mapping = {
    (1, "A"): "Saline",
    (2, "A"): "CNO",
    (1, "B"): "CNO",
    (2, "B"): "Saline",
}

# Update the 'treatment' column
for index, row in df.iterrows():
    treatment_suffix = treatment_mapping.get((row["batch"], row["treatment_AB"]), "")
    df.at[index, "treatment"] = f"{row['treatment']}{treatment_suffix}"

# Save the updated metadatafile to an Excel file
df.to_excel("processed_data\experiment_metadata_P8-acute-col-upd.xlsx", index=False)
