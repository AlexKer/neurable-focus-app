#!/usr/bin/env python3
import pandas as pd
import plotly.express as px
import datetime
import streamlit as st


WIN_LEN = 4  # seconds


# 5s sliding window, calculate variance between datapoints, plot
def rec_variances(df, threshold=0.005):
    rows_list = []

    # extract all variance for single rec
    window_size=75
    rec_attn_arr = df['y_'].values
    rec_dt_arr = df['dt'].values
    last_idx = rec_attn_arr.shape[0]-window_size
    rec_var_list = []
    for i in range(0,last_idx):
        window_var = rec_attn_arr[i:i+window_size].var()
        mid_idx = (i + i+window_size) // 2
        window_dt = rec_dt_arr[mid_idx]
        row_dict = {'window_dt': window_dt,
                    'window_var': -window_var}
        rows_list.append(row_dict)
        rec_var_list.append(window_var)
            
    # create df with cols |rec_nm|window_dt(center of 5min window)|window_variance
    var_df = pd.DataFrame(rows_list)

    return var_df


def read_lbls_today(fp):
    lbl_df = pd.read_csv(fp, sep="\t")
    lbl_df.columns = ['timestamp', 'activity', 'focused', 'comments']
    lbl_df['timestamp'] = pd.to_datetime(lbl_df['timestamp'])
    return lbl_df


def plot_multimodal(rec_lbl_df, rec_var_df):
    # plot variance
    f_var = px.scatter(
        rec_var_df, 
        x = "window_dt", 
        y = "window_var",
        color = "window_var"
    )

    # plot label activity
    f_act = px.scatter(
        rec_lbl_df, 
        x = "timestamp", 
        y="activity", 
        hover_data=['timestamp', 'activity', 'comments'])
    #        category_orders = {"activity": lbl_sorted},
    #    )
    f_f = px.scatter(
        rec_lbl_df, 
        x = "timestamp", 
        y="focused", 
        hover_data=['timestamp', 'activity', 'focused', 'comments']
    )

    # plot binary label
    # f_bin = px.scatter(
    #     rec_var_df, 
    #     x = "window_dt", 
    #     y = "focused",
    #     color = "focused",
    #     opacity = 0.5, 
    # )
    # # inference timeseries

    # f_inf = px.scatter(
    #     infs_df, 
    #     x = "dt", 
    #     y = "y_"
    # )
    # f_inf.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    # f_inf.update_xaxes(matches=None, showticklabels=False)
    # f_inf.show()

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        subplot_titles=("Survey Responses", 
            "",
            "Machine Learning Prediction of EEG"),
        # specs = [{}, {}, {}],
        vertical_spacing = 0.10)
    # focus from low to high
    fig.add_trace(f_act.data[0], row=1, col=1)
    fig.add_trace(f_f.data[0], row=2, col=1)
    fig.add_trace(f_var.data[0], row=3, col=1)

    # categories of activity ranked from low to high
    fig.update_yaxes(title="Activity", row=1, col=1)
    fig.update_yaxes(title="Focus", row=2, col=1)

    #fig.update_yaxes(type='category',categoryarray=lbl_sorted, row=1, col=1)
    #fig.update_yaxes(type='category',categoryarray=['false','true'], row=3, col=1)
    fig.update_yaxes(title='attention stability', row=3, col=1)
    fig.update_yaxes(showticklabels=False, row=3, col=1)
    # hide colour gradient
    fig.update(layout_coloraxis_showscale=False)
    fig.show()
    
    st.plotly_chart(fig)


def append_computed_timestamps(x, rec_nm, meta_df):
    """Prepend timestamps to a dataframe based on reference start-time

    Parameters
    ----------
    x : pd.DataFrame
        data frame to prepend
    meta_df : pd.DataFrame
        lookup table for start times

    Returns
    -------
    pd.DataFrame
        one extra column on x 'ts' containing a datetime
    """
    # extract start time from EEG front matter
    datetime_start_str = meta_df.loc[meta_df["rec_nm"]=='dogfood_'+rec_nm]["datetime"].values[0]
    datetime_start = datetime.datetime.fromisoformat(datetime_start_str)

    # compute timestamps
    delta_ts = [datetime.timedelta(seconds = WIN_LEN * i) for i in range(x.shape[0])]

    # prepend to target table
    x = x.copy()
    _cols_orig = x.columns.values.tolist()
    x['dt'] = [datetime_start + dt for dt in delta_ts]
    x = x[['dt'] + _cols_orig]

    return x


if __name__ == "__main__":
    """Main demo title +/- description
    """
    import doctest
    doctest.testmod()

