"""View multimodal timeseries for a single session

Inputs
------
FP_INF: inference table for one session
FP_LBL: google form labels
FP_META: EEG recording metadata (start-timestamp)
"""
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

import view_utils


#%% Params
DIR = os.getcwd()
DIR_DATA = DIR+'/output/new'
FP_INF = DIR_DATA+'/jc_4_inf.csv' # inference model outputs
FP_LBLS = DIR_DATA+'/multivariate_lbls.tsv' # based on new gform
FP_META = DIR_DATA+'/meta.csv' # metadata

WIN_LEN = 4  # seconds
EXPERIMENT='dogfood'


# extract recording name from the inference file
_REC_NM = pathlib.Path(FP_INF).stem.replace("_inf", "")


#%% preconditions
# python 3.7 required for datetime fxnality
_py_version = float(sys.version[:3])
if _py_version < 3.7:
    raise RuntimeError(f"Python 3.7+ required. Got {_py_version}")

assert path.isfile(FP_INF)
assert path.isfile(FP_LBLS)
assert path.isfile(FP_META)


#%% input interface
# EEG data front matter
meta_df = pd.read_csv(FP_META)
meta_df.columns = ['rec_nm', 'datetime', 'device', 'samp_rate']


# inference
# now single inf file
infs_df = pd.read_csv(FP_INF)


# labels 
lbl_df=view_utils.read_lbls_today(FP_LBLS)
print(lbl_df.head())

# TODO: filter to only current recording

# # sorted labels, focused -> distracted s.t. it matches with low var -> high var
# lbl_ord = ['Eyes Closed 5 mins', 'Focused', 'Typing', 'Talking',
#        'Drinking water/coffee', 'Drinking water/coffee/eating',
#         'Not focused/Distracted', 'Moving around', 'Walking']
# lbl_ord.reverse()
# k = lambda i: lbl_ord.index(i)
# sort lbl_df according to lambda
# lbl_sorted = sorted(lbl_df['activity'].unique(), key=k)


#%% Main
#! Singleton infs_df has 1 col "y_"
# append computed timestamps (based on front matter)
infs_df = view_utils.append_computed_timestamps(infs_df, _REC_NM, meta_df)


#%% Views

#%% Single-type, multi-session





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%% 
var_df = view_utils.rec_variances(infs_df)



# # threshold list mapping from rec_num -> threshold
# threshold_dict = {}
# def get_threshold(arr):
#     p = np.percentile(arr, 50)
#     return p
# # add additional binary focus col depending on rec threshold
# print(threshold_dict)
# var_df['threshold'] = var_df['rec_nm'].map(threshold_dict)
# var_df['focused'] = var_df['window_var'] > var_df['threshold']

# threshold_dict[rec_nm] = -get_threshold(rec_var_list)



#%% Single-session, multi-modal
st.title("**Neurable Workday Analysis**")
st.write("How to use: 1. Click and drag to select region and zoom in. 2. Double click to zoom out.")


# # inference timeseries
# f_inf = px.scatter(
#     infs_df, 
#     x = "dt", 
#     y = "y_"
# )
# f_inf.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# f_inf.update_xaxes(matches=None, showticklabels=False)
# f_inf.show()

# plot variance
f_var = px.scatter(
    var_df, 
    x = "window_dt", 
    y = "window_var",
    color = "window_var"
)
f_var.show()


# plot label activity
f_act = px.scatter(
    lbl_df, 
    x = "timestamp", 
    y="activity", 
    hover_data=['timestamp', 'activity', 'focused', 'comments']
)
f_act.show()


f_f = px.scatter(
    lbl_df, 
    x = "timestamp", 
    y="focused", 
    hover_data=['timestamp', 'activity', 'focused', 'comments']
)
f_f.show()
#    category_orders = {"activity": lbl_sorted}

#%%
view_utils.plot_multimodal(lbl_df, var_df)
#%%