import pandas as pd
import os
import re
import json


def create_metadata_excel(edf_folder, subject_metadata_file):

    smeta = pd.read_excel(subject_metadata_file, dtype={'mouseName':str, 'mouseId':str})
    df = pd.DataFrame()

    for i, edf in enumerate(os.listdir(edf_folder)):
        if not edf.endswith(".edf"):
          continue
        print(edf)
        # Parse file name
        temp_name, ext = os.path.splitext(edf) # TODO tell VAS: add split.ext
        
        info = re.split('_', temp_name)
        transmitterId = info[1]
        batch = str(info[2])
        day = str(info[3])
        subjectId = str(info[4])
        # surgery = str(info[5]) # get surgery from metadatafile
        injection = str(info[6])
        date = info[7]
        time = info[8]
        sesId = info[9]

        minfo = smeta[smeta['mouseId']==subjectId]
        tmp = pd.DataFrame({
            'edf' : edf,
            'date' : date,
            'time' : time,
            'sesId' : sesId,
            'transmitterId' : transmitterId,
            'mouseId' : subjectId,
            'batch' : batch,
            'day' : day,
            'injection' : injection,
            'surgery': minfo['surgery'].tolist()[0],
            'rfid': minfo['RFID'].tolist()[0],
            'cage': minfo['cage'].tolist()[0],
            'sex': minfo['sex'].tolist()[0],
            'arena': minfo['arena'].tolist()[0],
            'arena_position': minfo['arena_position'].tolist()[0],
            'species': minfo['species'].tolist()[0]

        }, index=[i])

        df = pd.concat([df, tmp])
    return df

if __name__ == '__main__':
    # Load settings
    with open('taini_colonies-main/settings.json', "r") as f:
        settings = json.load(f)

    create_metadata_excel(settings['edf_folder'], settings['subject_metadata']).to_excel(settings['metadata'], index=False)