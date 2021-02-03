import os
import re
import yaml
import pandas as pd

from . import settings


def read_cognionics_data(fp, **kwargs):
    """Read cognionics data file
    
    Notes
    -----
    !!! DUPLICATED FROM `proto/unsupervised.ipynb`

    This requires ~1m per 1h of data to run.

    Cognionics txt data scheme:
    First file line: header w/ file format
    Lines 3-5: YAML-formated metadata
    Line 6: Timeseries Column names (Tab-delimited)
    Line 7-: data
    """
    EEG_CHANNEL_COLNM_REGEX=r"Channel [0-9]+\(uV\)"

    # Predicate for whether column names are index or eeg channels (vs other False)
    get_index_eeg = lambda c: c == 'Time (s)' or re.search(EEG_CHANNEL_COLNM_REGEX, c)  

    cgx_df = pd.read_csv(fp, skiprows=4, sep=r"\t", 
        engine='python', thousands=",", usecols = get_index_eeg, **kwargs)

    #! CLIP OFF INDEX; ASSUME 20 channels
    eeg_df = cgx_df.iloc[:, 1:21]

    #!work-around: CLIP CGX acquisition extra channel name suffix "(uV)" to look like Unity Charmeleon
    eeg_df.rename(columns=lambda s: s.replace("(uV)", ""), inplace=True)

    # Rename channels to 1020 naming
    from neurablegrunt.functions.io import get_mapper_dict
    mapper_dict = get_mapper_dict()
    ch_mapper = mapper_dict['CGX_20ch']
    eeg_df.rename(columns=ch_mapper, inplace=True)

    return eeg_df


def read_cognionics_header(fp):
    IsYamlLine = lambda line: re.search(r":", line)
    IsDataColNameLine = lambda line: re.search(r"^Time", line)

    yaml_front_matter_lines = []

    with open(fp, "r") as fi:
        line = fi.readline()
        while not IsDataColNameLine(line):
            if IsYamlLine(line):
                yaml_front_matter_lines.append(line)
            line = fi.readline()

    front_matter_dict = {}
    for l in yaml_front_matter_lines:
        front_matter_dict.update(yaml.load(l, Loader=yaml.FullLoader))
        
    return front_matter_dict


# Labels
FP_GFORM_SHEET=os.path.join(settings.DIR_LBL, "gform_lbls.tsv")

def read_gform_sheet(fp):
    lbl_df = pd.read_csv(fp, sep="\t")
    lbl_df.columns = ['timestamp', 'activity', 'comments', 'recording']
    lbl_df = lbl_df[['recording', 'timestamp', 'activity', 'comments']]
    lbl_df['timestamp'] = pd.to_datetime(lbl_df['timestamp'])
    return lbl_df


if __name__=="__main__":
    import os.path as path
    assert path.isfile(FP_GFORM_SHEET)

    lbl_df = read_gform_sheet(FP_GFORM_SHEET)

    import plotly.offline as pyo
    pyo.init_notebook_mode()
    import plotly.express as px

    # plot label timeseries
    px.scatter(
        lbl_df, 
        x = "timestamp", 
        y = "recording", 
        color="activity", 
        hover_data=['recording', 'timestamp', 'activity', 'comments']
    )