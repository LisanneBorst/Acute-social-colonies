import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils import find_behaviors_measures, calc_plot_dimensions, highlight_days

# Define these
data_path = "processed_data\processed_data_P8-acute-col_with-metadata.xlsx"
plot_out_path = "plots"
experiment_name = "P8-acute-col"
group_var = "treatment"
x_axis_var = "day_bin"  # Usually session_bin
scale_factor = 1  # Scale the text of the final figure

# Stuff for plotting
custom_palette = {
    "CNO": "#118691",
    "Saline": "#a5275e",
}

# Select either one to subset
surgery = "DREADDs"
# surgery = "mCherry"

# Exclude dictionary where key:val is column:[values to be excluded] (array-like)
excludes = {
    "Video Name": [
        "Cage1_Acclimatization.SES",
        "Cage2_Acclimatisation.SES",
        "CSIVideo_2023_5_8_16_1_12.SES",
    ],
    "remarks": ["slow after inj!"],
}


def main():
    df = pd.read_excel(data_path, dtype={"animal_id": str})

    # Strip the spaces
    df[group_var] = df[group_var].str.strip()

    # Filtering out rows based on exclusion criteria
    for c, v in excludes.items():
        df = df[~df[c].isin(v)]

    # Subset surgery groups
    df = df[df["surgery"] == surgery]
    print(df["surgery"].drop_duplicates())
    print(df[["treatment", "animal_id"]].drop_duplicates())

    # Extract all available behaviors and corresponding measures
    behaviors_measures = find_behaviors_measures(df)

    # determine plot dimensions
    nrows = len(behaviors_measures)
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 3.5 * nrows))
    ax = ax.ravel()

    for i, (beh, measure) in enumerate(behaviors_measures):
        # Slice data
        df_beh = df[df["behavior"] == beh].copy()

        # Plot
        sns.lineplot(
            data=df_beh,
            x=x_axis_var,
            y=measure,
            ax=ax[i],
            hue=group_var,
            errorbar="se",
            palette=custom_palette,
        )

        ax[i].set_ylabel(measure)
        ax[i].set_xlabel("Time bin (h)")
        ax[i].set_title(f"{surgery} {beh} {measure}")
        ax[i].set_xticks(np.arange(0, df_beh[x_axis_var].max(), 12))
        ax[i].legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
        ax[i].margins(x=0)

        highlight_days(
            df_beh[x_axis_var].max(),
            ax[i],
            dark_first=True,
            light_color="#f2f2f2",
            dark_color="#999999",
        )

    for text_obj in plt.findobj(match=plt.Text):
        text_obj.set_fontsize(text_obj.get_fontsize() * scale_factor)

    plt.tight_layout()

    # Save
    os.makedirs(plot_out_path, exist_ok=True)
    plt.savefig(f"{plot_out_path}/all_lineplots_{surgery}_{experiment_name}.pdf")
    plt.savefig(f"{plot_out_path}/all_lineplots_{surgery}_{experiment_name}.png")


if __name__ == "__main__":
    main()
