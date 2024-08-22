import numpy as np
import re
import pandas as pd

def str_sync_to_array(s):
    iv = int(re.split("_", s)[1])
    return np.array([bool(iv & 2**i) for i in range(12)]).astype(int).tolist()


def load_event_trace(filepath):
    '''
    Loads and cleans the event trace file
    '''
    # Load event trace
    if 'Book1' in filepath:
        day1 = pd.read_excel(filepath, sheet_name=None, skiprows=np.arange(0,6))
        path2 = re.sub(r'Book1', 'Book2', filepath)
        try:
            day2 = pd.read_excel(path2, sheet_name=None, skiprows=np.arange(0,6))
            # add these dictionaries together
            day1.update(day2)
            print('Found two excels')
        except FileNotFoundError:
            pass
    elif 'Book2' in filepath:
        print("Starting with Book2")
        day2 = pd.read_excel(filepath, sheet_name=None, skiprows=np.arange(0,6))
        path2 = re.sub(r'Book2', 'Book1', filepath)
        try:
            day1 = pd.read_excel(path2, sheet_name=None, skiprows=np.arange(0,6))
            # add these dictionaries together
            day1.update(day2)
            print('Found two excels')
        except FileNotFoundError:
            pass
    else: # only one book
        print('Only one excel')
        day1 = pd.read_excel(filepath, sheet_name=None, skiprows=np.arange(0,6))
    
        
    # Add all trials into one dataframe
    df1 = pd.DataFrame()

    for trial, df in day1.items():
        id = re.search('Trial (\d+)', trial)[1]

        df = df.drop(['ID'], axis=1)
        df['Trial'] = int(id)
        df = df.rename(columns = {'From Frame':'start_frame', 'To Frame': 'end_frame', 'Length(Frame)': 'length'})
        df1 = pd.concat([df1, df])
    
    # Add the "Day" column
    df = df1.reset_index(drop=True)
    # Find the first trial number of a different day
    day_start_trials = [1] # Have the first trial number in there by default
    for index, row in df.iterrows():
        if (index>0 and row['start_frame'] < df.loc[index-1, 'start_frame']) and (row['Trial'] != df.loc[index-1, 'Trial']):
            day_start_trials.append(row['Trial'])

    final_trial = df['Trial'].max()
    # Add the "Day" information to the df
    df2 = df1.copy()
    df2['Day'] = None

    for day_count, start_day in enumerate(day_start_trials):
        try:
            day_range = np.arange(start_day, day_start_trials[day_count+1])
        except IndexError:
            day_range = np.arange(start_day, final_trial+1)

        df2.loc[df2['Trial'].isin(day_range), 'Day'] = day_count + 1

    return df2