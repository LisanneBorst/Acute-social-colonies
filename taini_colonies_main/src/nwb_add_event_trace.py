import numpy as np
from pynwb.behavior import IntervalSeries, BehavioralEpochs
import pandas as pd
from pynwb import NWBHDF5IO
from ndx_events import LabeledEvents, AnnotatedEventsTable, TTLs
import os
import re
import json
from datetime import date
from taini_colonies_utils import load_event_trace
from nwb_data_retrieval_functions import get_arena_position, get_day, get_animal_id

if __name__ == "__main__":

    # Load settings
    with open("settings.json", "r") as f:
        settings = json.load(f)

    nwb_folder = settings["nwb_files_folder"]
    behavior_folder = settings["event_trace_folder"]

    # Define these so we can extract behaviors of interest
    # ALWAYS ASSUMING THAT ANIMAL OF INTEREST IS No 1
    # behaviors_patterns = {
    #     'social_contact' : r'Social Contact \[? ?1(\[\#\d+\])? with \d+(\[\#\d+\])? ?\]?',
    #     'social_approach' : r'Social Approach \[? ?1(\[\#\d+\])? to \d+(\[\#\d+\])? ?\]?',
    #     'social_follow' : r'Social Follow \[? ?1(\[\#\d+\])? follow \d+(\[\#\d+\])? ?\]?',
    #     'social_sniff' : r'Social Sniff \[? ?1(\[\#\d+\])? sniff \d+(\[\#\d+\])? ?\]?',
    #     'social_leave' : r'Social Leave \[? ?1(\[\#\d+\])? from \d+(\[\#\d+\])? ?\]?',
    #     'social_hide' :  r'Social Hide \[? ?1(\[\#\d+\])? with \d+(\[\#\d+\])? ?\]?',
    #     'contact_nest' : r'Animal 1.* Contact Nest',
    #     'hide_in_nest' : r'Animal 1.* Hide in Nest'
    # }

    # Patterns
    behaviors_patterns = {
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
    }  # TODO AREA

    # Main loop
    for nwb_file in os.listdir(nwb_folder):
        with NWBHDF5IO(f"{nwb_folder}/{nwb_file}", "a") as io:
            print(f"Processing NWB file: {nwb_file}")
            nwb = io.read()
            # Skip if there is an event trace already in there
            if nwb.processing:
                print(f"Skipping {nwb_file} (processing module already exists)")
                continue

            animal_id = get_animal_id(f"{nwb_folder}/{nwb_file}")
            arena_position = get_arena_position(f"{nwb_folder}/{nwb_file}")
            day = (
                int(get_day(f"{nwb_folder}/{nwb_file}")) + 1
            )  # +1 bc acclimatization day was added as well

            # extra check for the day
            print(f"{nwb_file} has day: {day}")

            # Find corresponding event trace file
            vnum = None
            for events_file in os.listdir(behavior_folder):
                if animal_id in events_file:
                    events_df = load_event_trace(f"{behavior_folder}/{events_file}")
                    print(f"events_df for {events_file}: {events_df}")
                    # Get event trace version number
                    vnum = re.search("event-trace-v(\d+)", events_file)[1]
                    break
            if vnum == None:
                print(f"No event trace for {nwb_file}")
                continue

            # Create behavior processing module
            all_epochs = BehavioralEpochs(name="all_colony_behaviors")
            behavior_module = nwb.create_processing_module(
                name=f"behavior_v{vnum}_{date.today().isoformat()}",
                description=f"colony event trace v{vnum}",
            )

            # Extract behaviors
            for behavior, pattern in behaviors_patterns.items():
                print(f"Selected behavior: {behavior}")
                pattern = pattern.format(arena_position)
                print(f"pattern: {pattern}")

                d = events_df[
                    (events_df["Event"].str.contains(pattern))
                    & (events_df["Day"] == int(day))
                ]
                print(f"d created: {d}")

                start_timestamps = d["start_frame"].to_numpy()
                end_timestamps = d["end_frame"].to_numpy()
                print(f"start_timestamps for {behavior}: {start_timestamps}")
                print(f"end_timestamps for {behavior}: {end_timestamps}")

                # Create timestamps and the corresponding events(aka 1 or -1 for start/stop)
                timestamps = np.empty(int(start_timestamps.shape[0] * 2))
                events = np.empty(int(start_timestamps.shape[0] * 2))
                timestamps[::2] = start_timestamps
                timestamps[1::2] = end_timestamps

                events[::2] = 1
                events[1::2] = -1

                timestamps = timestamps.astype(int)
                print(f"final timestamps for {behavior}: {timestamps}")
                events = events.astype(int)
                print(f"final events for {behavior}: {events}")

                # Create Behavioral Epoch object
                behavioral_epochs = IntervalSeries(
                    name=behavior,
                    description=behavior,
                    data=events,
                    timestamps=timestamps,
                )

                # Append to epochs object
                all_epochs.add_interval_series(behavioral_epochs)
            # Append all epochs to the nwb file
            behavior_module.add(all_epochs)
            io.write(nwb)
            print(f"Behavioral data added to NWB file: {nwb_file}")
