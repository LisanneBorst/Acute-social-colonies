import numpy as np
from pynwb.behavior import IntervalSeries, BehavioralEpochs, Position, SpatialSeries
import pandas as pd
from pynwb import NWBHDF5IO, TimeSeries
from ndx_events import LabeledEvents, AnnotatedEventsTable, TTLs
import os
import re
import json
from .taini_colonies_utils import load_event_trace


def make_coordinates_df(xy_folder, trial_range):
    '''
        Creates a clean dataframe with information all the relevant
        spatial information. 
        
        Args: path, of the folder that contains all the .TXT files 
            containing the raw coordinate data
    '''

    # Find how many animals in the recording session
    file = os.path.join(xy_folder, os.listdir(xy_folder)[0])
    df = pd.read_csv(file, delimiter = '\t', skiprows=np.arange(0, find_table_start(file)))
    max_animals = df.columns[df.columns.str.contains('CenterY\(mm\)')].to_list().__len__()

    # Find corresponding filepaths
    coordinate_paths = []
    for trial in trial_range:
        for f in os.listdir(xy_folder):
            if re.search(f'_{trial}_TCR.TXT', f):
                coordinate_paths.append(os.path.join(xy_folder, f))

    # Make DataFrame
    df2 = pd.DataFrame()
    for file in coordinate_paths:
        df = pd.read_csv(file, delimiter = '\t', skiprows=np.arange(0, find_table_start(file)))
        for i in range(max_animals):
            if i == 0:
                df2 = pd.concat([
                    df2,
                    pd.DataFrame({
                        'Animal': i+1,
                        'Frame': df['Format:FrameNum'].to_numpy(),
                        'CenterX': df['CenterX(mm)'].to_numpy(),
                        'CenterY': df['CenterY(mm)'].to_numpy(),
                        'NoseX': df['NoseX(mm)'].to_numpy(),
                        'NoseY': df['NoseY(mm)'].to_numpy(),
                        'Motion': df['Motion'].to_numpy(),
                        'Orientation': df['Orientation(-pi/2 to pi/2)'].to_numpy()

                    })
                ])
            
            else:
                df2 = pd.concat([
                    df2,
                    pd.DataFrame({
                        'Animal': i+1,
                        'Frame': df['Format:FrameNum'].to_numpy(),
                        'CenterX': df[f'CenterX(mm).{i}'].to_numpy(),
                        'CenterY': df[f'CenterY(mm).{i}'].to_numpy(),
                        'NoseX': df[f'NoseX(mm).{i}'].to_numpy(),
                        'NoseY': df[f'NoseY(mm).{i}'].to_numpy(),
                        'Motion': df[f'Motion.{i}'].to_numpy(),
                        'Orientation': df[f'Orientation(-pi/2 to pi/2).{i}'].to_numpy()

                    })
                ])
    return df2.reset_index(drop=True)


def get_arena_bbox(xy_file):
    '''
        Returns a str with the top,bottom, left, right coordinates
        of the arena. It is used to be added as a description to the 
        pynwb.behavior.SpatialSeries
        
        Args: path, of a single file
    '''
    with open(xy_file, 'r') as f:
        d = f.readlines()
        for line in d:
            if 'Arena Bounding Box' in line:
                left, right, top, bottom = re.findall(r"Left :(\d+)\t Right :(\d+)\t Top :(\d+)\t Bottom :(\d+)", line)[0]
                left, right, top, bottom = (int(i) for i in [left, right, top, bottom])
                break
        
    return f'left: {left}, right: {right}, top: {top}, bottom: {bottom}'

def find_table_start(xy_file):
    '''
        Finds the txt line index where the data table starts

        Args: path, of a single file
        Returns: int
    '''
    with open(xy_file, 'r') as f:
        d = f.readlines()
        for i, (line) in enumerate(d):
            if 'Format:' in line:
                return i
        raise ValueError('Cannot find coordinate data table start. Does it start with "Format:" ?')



if __name__ == '__main__':

    # Load settings
    with open('settings.json', "r") as f:
        settings = json.load(f)

    nwb_folder = settings['nwb_files_folder']
    behavior_folder = settings['event_trace_folder']
    coordinates_folder = settings['coordinate_data_folder']

    # Main Loop
    for nwb_file in os.listdir(nwb_folder):
        with NWBHDF5IO(f'{nwb_folder}/{nwb_file}', "a") as io:
            print(nwb_file)
            nwb = io.read()
            animal_id = nwb.subject.subject_id
            day = re.split('_', nwb.identifier)[2]
            day = re.search('\d', day)[0]
            print(animal_id, day)

            # Find corresponding event trace file
            for events_file in os.listdir(behavior_folder):
                if animal_id in events_file:
                    events_df = load_event_trace(f'{behavior_folder}/{events_file}')
                    break

            # Find trial range for this recording day
            trial_range = events_df[events_df['Day']==int(day)]['Trial'].unique()
            print(trial_range)

            # Parse the coordinate data
            # Find xy_folder for this animal

            if animal_id not in os.listdir(coordinates_folder):
                print(f'No coordinates folder found for animal {animal_id}')
                continue
            else:
                for animal_folder in os.listdir(coordinates_folder):
                    if animal_id in animal_folder:
                        xy_folder = os.path.join(coordinates_folder, animal_folder)
                        print(coordinates_folder, animal_folder, xy_folder)
                        # Also get arena Bbox
                        arena_reference = get_arena_bbox(os.path.join(xy_folder, os.listdir(xy_folder)[0])) 
                        break
                xy_data = make_coordinates_df(xy_folder, trial_range)
            
            # Make new behavioral module
            behavior_module = nwb.create_processing_module(
                name="coordinate_data", description="Raw coordinate/motion/head orientation data"
            )
            # Make SpatialSeries and TimeSeries (motion/orientation) objects for all subjects
            for a in xy_data['Animal'].unique():
                spatial_series_center = SpatialSeries(
                    name = f"xy_center_{a}",
                    description = f"(x,y) center position in colony for the animal ({a})",
                    data = xy_data[xy_data['Animal']==a][['CenterX', 'CenterY']].to_numpy(),
                    timestamps = xy_data[xy_data['Animal']==a]['Frame'].to_numpy(),
                    reference_frame = arena_reference,
                    unit='mm'
                )

                spatial_series_nose = SpatialSeries(
                    name = f"xy_nose_{a}",
                    description = f"(x,y) nose position in colony for the animal ({a})",
                    data = xy_data[xy_data['Animal']==a][['NoseX', 'NoseY']].to_numpy(),
                    timestamps = xy_data[xy_data['Animal']==a]['Frame'].to_numpy(),
                    reference_frame = arena_reference,
                    unit='mm'
                )

                motion_series = TimeSeries(
                    name = f"motion_{a}",
                    description = f"motion in colony for the experimental animal ({a})",
                    data = xy_data[xy_data['Animal']==a][['Motion']].to_numpy(),
                    timestamps = xy_data[xy_data['Animal']==a]['Frame'].to_numpy(),
                    unit = 'N/A',
                    comments = 'The timestamps refer to the frame number'
                )

                orientation_series = TimeSeries(
                    name = f"orientation_{a}",
                    description = f"head orientation in colony for the animal ({a})",
                    data = xy_data[xy_data['Animal']==a][['Motion']].to_numpy(),
                    timestamps = xy_data[xy_data['Animal']==a]['Frame'].to_numpy(),
                    unit = 'pi ratio',
                    comments = 'The timestamps refer to the frame number'
                )
                
                behavior_module.add(spatial_series_center)
                behavior_module.add(spatial_series_nose)
                behavior_module.add(motion_series)
                behavior_module.add(orientation_series)
            io.write(nwb)
            print(f'Added spatial info to {nwb_file}')

