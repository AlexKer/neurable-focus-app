import os
import sys
import os.path as path
import glob
import datetime
import pathlib
import pandas as pd
import numpy as np

import plotly.offline as pyo
pyo.init_notebook_mode()
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import streamlit as st
from utils import settings

#%% Params
DIR_DATA = settings.DIR_WORKDAY+'/output'
FP_LBLS = settings.DIR_LBL+'/gform.tsv'
WIN_LEN = 4  # seconds


#%% preconditions
# python 3.7 required for datetime fxnality
_py_version = float(sys.version[:3])
if _py_version < 3.7:
    raise RuntimeError(f"Python 3.7+ required. Got {_py_version}")


assert path.isdir(DIR_DATA)
assert path.isfile(FP_LBLS)


#%% input interface
os.listdir(DIR_DATA)

# EEG data front matter
meta_df = pd.read_csv(path.join(DIR_DATA, "meta.csv"))
meta_df.columns = ['rec_nm', 'datetime', 'device', 'samp_rate']

# inference
inf_fps = glob.glob(DIR_DATA+"/*_inf.csv")

# labels
lbl_df = pd.read_csv(FP_LBLS, sep="\t")
lbl_df.columns = ['timestamp', 'activity', 'comments', 'recording']
lbl_df = lbl_df[['recording', 'timestamp', 'activity', 'comments']]
lbl_df['timestamp'] = pd.to_datetime(lbl_df['timestamp'])

# sorted labels, focused -> distracted s.t. it matches with low var -> high var
lbl_ord = ['Eyes Closed 5 mins', 'Focused', 'Typing', 'Talking',
       'Drinking water/coffee', 'Drinking water/coffee/eating',
        'Not focused/Distracted', 'Moving around', 'Walking']
lbl_ord.reverse()
k = lambda i: lbl_ord.index(i)
lbl_marker_size_dict = {}
for i,v in enumerate(lbl_ord):
    lbl_marker_size_dict[v]=i
lbl_df['marker_size']=lbl_df['activity'].map(lbl_marker_size_dict)

#! TODO bads


#%% Fxns
def get_session_data(lbls, infs):
    assert lbls in lbl_df['recording'].unique(), f"Labels not found for {label_recording}.  Available labels: {lbl_df['recording'].unique()}"
    rec_lbl_df = lbl_df.loc[lbl_df['recording'] == lbls]
    if isinstance(infs, str):
        infs=[infs]
    rec_inf_df = var_df.loc[var_df['rec_nm'].isin(infs)]
    return rec_inf_df, rec_lbl_df


def plot_multimodal(inf_df, lbl_df):
    # sort lbl_df according to lambda
    lbl_sorted = sorted(lbl_df['activity'].unique(), key=k)

    # plot inference timeseries
    f_inf = px.scatter(
        inf_df, 
        x = "window_dt", 
        y = "window_var",
        color = "window_var"
    )

    # plot activity
    f_act = px.scatter(
        lbl_df, 
        x = "timestamp", 
        y="activity", 
        hover_data=['recording', 'timestamp', 'activity', 'comments'],
        category_orders = {"activity": lbl_sorted},
    )

    # plot binary Labels
    #inf_df["focused"] = inf_df["focused"].astype(str)
    f_bin = px.scatter(
        inf_df, 
        x = "window_dt", 
        y = "focused",
        color = "focused",
        opacity = 0.5, 
    )

    # f_inf.show()
    # f_act.show()
    # f_bin.show()

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        subplot_titles=("Your Activity Labels", 
            "Machine Learning Prediction of EEG",
            "Focused? (from our algorithm)"),
        # specs = [{}, {}, {}],
        vertical_spacing = 0.05)
    # focus from low to high
    fig.add_trace(f_act.data[0], row=1, col=1)
    fig.add_trace(f_inf.data[0], row=2, col=1)
    fig.add_trace(f_bin.data[0], row=3, col=1)
    # categories of activity ranked from low to high
    fig.update_yaxes(type='category',categoryarray=lbl_sorted, row=1, col=1)
    fig.update_yaxes(type='category',categoryarray=['false','true'], row=3, col=1)
    fig.update_yaxes(title='level of focus (calm)', row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    # hide colour gradient
    fig.update(layout_coloraxis_showscale=False)
    fig.show()
    
    st.plotly_chart(fig)


#%% Main
# Load inference tables and append computed timestamps (based on front matter)
inf_dfs = []
for fp in inf_fps:
    rec_nm = pathlib.Path(fp).stem.replace("_inf", "")
    inf_df = pd.read_csv(fp)

    # extract start time from EEG front matter
    datetime_start_str = meta_df.loc[meta_df["rec_nm"]==rec_nm]["datetime"].values[0]
    datetime_start = datetime.datetime.fromisoformat(datetime_start_str)

    # append computed times to inference table
    delta_ts = [datetime.timedelta(seconds = WIN_LEN * i) for i in range(inf_df.shape[0])]
    inf_df['dt'] = [datetime_start + dt for dt in delta_ts]

    # add rec_nm identifier; pretty
    inf_df.columns = ['attn', 'dt']
    inf_df['rec_nm'] = rec_nm
    inf_df = inf_df[['rec_nm', 'dt', 'attn']]

    inf_dfs.append(inf_df)

inf_df = pd.concat(inf_dfs)

# append recording lengths to meta_df (based on No. of inference scores)
eeg_lens = {}
for nm, gdf in inf_df.groupby(['rec_nm']):
    eeg_lens[nm] = gdf.shape[0] * WIN_LEN / 60
len_df = pd.DataFrame.from_dict(eeg_lens, orient='index').reset_index()
len_df.columns = ['rec_nm', 'rec_len']

meta_df = pd.merge(meta_df, len_df, on='rec_nm', how="outer")

#%% Tables
meta_df.sort_values('rec_nm')


#%% Views

#%% Single-type, multi-session

# inference timeseries
f_inf = px.scatter(
    inf_df, 
    x = "dt", 
    y = "attn",
    facet_row = "rec_nm"
)
f_inf.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
f_inf.update_xaxes(matches=None, showticklabels=False)
f_inf.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5s sliding window, calculate variance between datapoints, plot

def get_all_rec_variances(df, threshold=0.005):
    rec_nm_list = df['rec_nm'].unique()
    rows_list = []
    # extract all variance for single rec
    def extract_windows_variance(rec_nm, rec_df, window_size=75):
        rec_attn_arr = rec_df['attn'].values
        rec_dt_arr = rec_df['dt'].values
        custom_threshold = rec_attn_arr.mean()
        last_idx = rec_attn_arr.shape[0]-window_size
        for i in range(0,last_idx):
            window_var = rec_attn_arr[i:i+window_size].var()
            mid_idx = (i + i+window_size) // 2
            window_dt = rec_dt_arr[mid_idx]
            row_dict = {'rec_nm':rec_nm,
                        'window_dt': window_dt,
                        'window_var': -window_var,
                        'focused': window_var < custom_threshold}
            rows_list.append(row_dict)
    # loop through all recs
    for rec_nm in rec_nm_list:
        rec_df = df[df['rec_nm']==rec_nm]
        extract_windows_variance(rec_nm, rec_df)
    # create df with cols |rec_nm|window_dt(center of 5min window)|window_variance
    var_df = pd.DataFrame(rows_list)
    return var_df
var_df = get_all_rec_variances(inf_df)
print(var_df.head())



#%% Single-session, multi-modal
st.title("**Neurable Workday Analysis**")
st.write("How to use: 1. Click and drag to select region and zoom in. 2. Double click to zoom out.")
# AJ
# plot_multimodal(*get_session_data("AJ_dogfood_01212021", ["aj_1", "aj_2"]))
# Jegan
# plot_multimodal(*get_session_data("JC_DogFood_01202021", ["jc_1", "jc_2"]))
plot_multimodal(*get_session_data("JC_dogfood_01222021", ["jc_3"]))
# plot_multimodal(*get_session_data("JC_dogfood_01222021", ["jc_4"]))
# Ramses
# plot_multimodal(*get_session_data("RA_DogFood_01212022", ["ra_3"]))
# plot_multimodal(*get_session_data("RA_DogFood_01212021", ["ra_1", "ra_2"]))
# %%