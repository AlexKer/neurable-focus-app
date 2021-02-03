"""Collect metadata from CGX data file headers"""
import os.path as path
import glob
import pathlib
import pandas as pd

from utils import settings
from utils import fio


#%% Params
DIR_DATA_IN=settings.DIR_EEG
FP_OUT=path.join(settings.DIR_WORKDAY, "output", "meta.csv")


#%% Main
data_fps = glob.glob(DIR_DATA_IN + "/*.txt")
print(f"Parsing front matter from {len(data_fps)} raw eeg files: {data_fps}")

headers = []
for fp in data_fps:
    headers.append(fio.read_cognionics_header(fp))

# enframe front matter
data_file_stems = [pathlib.Path(fp).stem for fp in data_fps]
meta_df = pd.DataFrame.from_records(headers, index = data_file_stems)

# output interface
print(f"Saving metadata to {FP_OUT}")
meta_df.to_csv(FP_OUT)
