import re
import math
from datetime import datetime, timedelta
import warnings


def get_batch_colony(file):
    info = re.split("_", file)
    batch = None
    arena = None
    for part in info:
        if re.search(r"b\d+", part):
            batch = re.search(r"b(\d+.?)", part)[1]
        elif re.search(r"batch\d+", part):
            batch = re.search(r"batch(\d+.?)", part)[1]
        elif re.search(r"c\d+", part):
            arena = re.search(r"c(\d+)", part)[1]
        elif re.search(r"arena\d+", part):
            arena = re.search(r"arena(\d+)", part)[1]
        elif re.search(r"cage\d+", part):
            arena = re.search(r"cage(\d+)", part)[1]

    if (batch is None) or (arena is None):
        raise ValueError(f"The batch is {batch} and arena is {arena}")
    else:
        return batch, arena


# Function to extract mouse identifier from the 'Measure' column
def extract_mouse_id(measure_str):
    return measure_str.split("-")[0]


def calculate_day_bins_and_counts(df, unknown_date=True):

    if not unknown_date:
        # Sort data by 'real_date' and 'Trial ID' to ensure correct counting
        df.sort_values(by=["Trial ID", "real_date"], inplace=True)

        # 'day_bin': Counting the number of entries per day for each Trial ID
        df["day_bin"] = df.groupby(["Measure", "real_date"]).cumcount() + 1

        # 'day_count': Counting the number of distinct days for each Trial ID
        df["day_count"] = df["real_date"].rank(method="dense").astype(int)

        # Add total session bin (leave empty trials as gaps)
        df["session_bin"] = df["day_bin"] + (df["day_count"] - 1) * 24

        return df.reset_index(drop=True)
    else:
        print("Warning: Default video name is False, so real_date is probably wrong.")

        df.sort_values(by=["Trial ID"], inplace=True)

        # 'day_bin': Counting the number of entries per Video Name for each Trial ID
        df["day_bin"] = df.groupby(["Measure", "Video Name"]).cumcount() + 1

        # 'day_count': Counting the number of distinct days for each Trial ID
        df["day_count"] = df["Video Name"].rank(method="dense").astype(int)

        # Add total session bin (leave empty trials as gaps)
        df["session_bin"] = df["day_bin"] + (df["day_count"] - 1) * 24

        return df.reset_index(drop=True)


def extract_time_from_video_name(video_name):
    datetime_parts = video_name.split("_")[1:6]
    datetime_formats = ["%Y_%m_%d_%H_%M_%S", "%Y_%m_%d_%H_%M", "%Y_%m_%d_%H"]

    for fmt in datetime_formats:
        try:
            datetime_str = "_".join(datetime_parts[: len(fmt.split("_"))])
            datetime_obj = datetime.strptime(datetime_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unable to parse datetime from {video_name}")

    # Round to nearest hour
    if datetime_obj.minute >= 30:
        datetime_obj += timedelta(hours=1)

    return datetime_obj.replace(minute=0, second=0, microsecond=0)


def process_measures(bin_df):
    """
    bin_df should be cleaned
    """

    # For default video names we can extract date and time
    if all(bin_df["Video Name"].str.contains("CSIVideo")):
        bin_df["real_time"] = bin_df["Video Name"].apply(extract_time_from_video_name)
        bin_df["real_date"] = bin_df["real_time"].dt.date
        unknown_date = False
    # For other video names time and date is None, if needed add different methods to extract datetime info from the videoname
    else:  # Other naming
        bin_df["real_time"] = None
        bin_df["real_date"] = None
        unknown_date = True

    # add day bins
    return calculate_day_bins_and_counts(bin_df, unknown_date)


def get_exported_behaviors(measures):
    """
    Input should be a numpy array (normally a unique array of the 'Measures' columns of the Behavioral dataframe)
    Returns a list of all the measures that are found in the input

    Also prints unmatched events (in which case update the behavior_and_regex dict)

    Args:
        measures (array-like): An array of measures, usually unique values from a DataFrame column.

    Returns:
        list: A list of behavior labels that have matching patterns in the input measures.
    """

    behavior_and_regex = {
        "social_contact": r"Social Contact",
        "social_approach": r"Social Approach",
        "social_passive_approach": r"Social Approach",
        "social_follow": r"Social Follow",
        "social_passive_follow": r"Social Follow",
        "social_sniff": r"Social Sniff",
        "social_passive_sniff": r"Social Sniff",
        "hide_in_nest": r"Hide in Nest",
        "contact_nest": r"Contact Nest",
        "social_hide": r"Social Hide",
        "social_leave": r"Social Leave",
        "in_area": r"Area:.+In",
    }
    unique_measures = set(measures)
    matched_labels = set()
    unmatched_measures = set(unique_measures)

    for measure in unique_measures:
        for label, pattern in behavior_and_regex.items():
            if re.search(pattern, measure):
                matched_labels.add(label)
                unmatched_measures.discard(measure)

    # Handling multiple labels for the same pattern
    for label, pattern in behavior_and_regex.items():
        if any(re.search(pattern, measure) for measure in measures):
            matched_labels.add(label)

    if len(unmatched_measures) != 0:
        print("The following measures do not have regex patterns!")
        print(unmatched_measures)

    return list(matched_labels)


def find_max_animals(bin):
    """
    Finds max number of animals in exported data based on the Bin
    record (aka Distance moved or Social distance)

    Args:
        bin: DataFrame for bin measures
    Returns:
        max_animals: int
    """
    max_animals = bin["Measure"].str.extract(r"Mouse(\d+).+").max()[0]
    if max_animals != None:
        return int(max_animals)
    # Just in case something goes wrong return the default max_animals
    warnings.warn("max_animals could not be determined. Defaults to 4", UserWarning)
    return 4


def find_behaviors_measures(df):
    """
    Extracts combinations of behavior labels and their corresponding measures from a DataFrame.

    Parameters:
    df (DataFrame): A pandas DataFrame with columns 'behavior', 'bouts', 'duration', 'distance'.

    Returns:
    List[Tuple[str, str]]: A list of tuples where each tuple contains a behavior label and a measure.
    """
    # Extracting unique behaviors
    unique_behaviors = df["behavior"].unique()

    # Dictionary to store behavior and corresponding measures
    behavior_measures = {}

    # Loop through each behavior and extract corresponding measures
    for behavior in unique_behaviors:
        behavior_data = df[df["behavior"] == behavior]
        # Extracting measures and removing NaN values
        bouts = behavior_data["bouts"].dropna().unique().tolist()
        duration = behavior_data["duration"].dropna().unique().tolist()
        distance = behavior_data["distance"].dropna().unique().tolist()

        # Storing the measures in the dictionary
        behavior_measures[behavior] = {
            "bouts": bouts,
            "duration": duration,
            "distance": distance,
        }

    # Creating a list of tuples with ('behavior_label', 'measure')
    behavior_measure_tuples = []

    # Iterate over the dictionary to create tuples
    for behavior, measures in behavior_measures.items():
        # Add a tuple for each non-empty measure
        if measures["bouts"]:
            behavior_measure_tuples.append((behavior, "bouts"))
        if measures["duration"]:
            behavior_measure_tuples.append((behavior, "duration"))
        if measures["distance"]:
            behavior_measure_tuples.append((behavior, "distance"))

    return behavior_measure_tuples


def calc_plot_dimensions(n):
    """
    Calculate an optimized number of rows and columns for plotting n subplots,
    considering a more balanced aspect ratio.

    Args:
    n (int): The number of subplots.

    Returns:
    tuple: A tuple containing the number of rows and columns (nrows, ncols).
    """
    for ncols in range(int(math.sqrt(n)), 0, -1):
        if n % ncols == 0:
            nrows = n // ncols
            return nrows, ncols
    # Fallback if no exact division is found
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    return nrows, ncols


def highlight_days(
    max_val, ax, dark_first=False, light_color="#fcefbd", dark_color="#d9d5e0"
):

    if dark_first:
        x = 0
        y = 12
    else:
        ax.axvspan(xmin=0, xmax=12, color=light_color, alpha=0.5, edgecolor=None)
        x = 12
        y = 24

    while x < max_val:
        ax.axvspan(xmin=x, xmax=y, color=dark_color, alpha=0.5, edgecolor=None)
        if x + 12 <= max_val:
            ax.axvspan(
                xmin=x + 12, xmax=y + 12, color=light_color, alpha=0.5, edgecolor=None
            )
        x += 24
        y += 24
