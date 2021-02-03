#!/usr/bin/env python3
"""Functions to convert EEG waveform data into features"""
import os.path as path
import logging
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

import mne
from mne.time_frequency import psd_array_multitaper
from mne.connectivity import spectral_connectivity


BANDS = {"Delta": [2, 4],
         "Theta": [4, 8],
         "Alpha": [8, 13],
         "Beta": [13, 30],
         "Gamma": [30, 45]}

SR = 500
WINDOW_LENGTH = 4  # s
WINDOW_OVERLAP = 0  # s; used w/ sliding windows
WINDOW_START = -1  # s; used w/ cued windows


def neurablescreen_wrapper(win_df:pd.DataFrame, interpolation_kwargs:dict) -> pd.DataFrame:
    """Applies neurablescreen interpolation to a single window of data

    Parameters
    ----------
    win_df : pd.DataFrame
        Data window
    interpolation_kwargs : dict
        kwargs for neurablescreen. See neurablescreen documentation

    Returns
    -------
    pd.DataFrame
        Interpolated data
    """
    # Try importing neurablescreen
    try:
        from fma_sivox_analysis import neurablescreen
    except:
        print('Neurablescreen is not installed. Try installing or set do_interp to False.')
        print('To install, run:')
        print('git clone git@maat.neurable.com:interactions/fma_sivox_analysis.git')
        print('cd fma_sivox_analysis')
        print('git checkout dev')
        print('git submodule update --init --recursive')
        print('pip install -e .')
        raise ValueError('Neurablescreen not installed.')

    # ch names and types
    ch_names = list(win_df.columns)
    ch_types = ['eeg'] * len(ch_names)

    # build raw object
    info = mne.create_info(sfreq=SR, ch_names=ch_names, ch_types=ch_types)
    data = win_df.values.transpose()
    raw = mne.io.RawArray(data=data, info=info, verbose=False)

    # build epochs and set montage
    events = np.array([[0,0,0]])
    epochs_orig = mne.EpochsArray(data=np.expand_dims(data, 0), info=info, events=events, verbose=False)
    _ = epochs_orig.set_montage('standard_1020', verbose=False) # Required for interpolation

    # Apply interpolation / rejection
    (epochs_interpolated, reject_dict) = neurablescreen.rejection_suite_noninteractive(
            raw=raw, epochs_orig=epochs_orig,
            rejection_mode='Vpeak2peak_manual',
            debug_plot=False, verbose=False,
            **interpolation_kwargs
    )

    # mask indicating bad epochs
    bads_mask = reject_dict['rejectlog'].labels

    # pack back into win_df
    data = np.squeeze(epochs_interpolated._data).transpose()
    win_df = pd.DataFrame(data, columns=ch_names, index=win_df.index)

    return (win_df, bads_mask)


#! IMPURE FXN USES MODULE GLOBALS
def eeg_features(eeg_df, feats="power", do_interp = True, thresh_reject=300):
    """Wrapper to compute power features for a full waveform

    Parameters
    ----------
    eeg_df : pd.DataFrame
        waveform data
    feats : str
        feature set {power, coherence}
    thresh_reject : 300
        Threshold to use for peak-to-peak bad channel rejection. By default,
        set to 300.

    Returns
    -------
    feat_df : pd.DataFrame

    !I.F.F do_interp
    feat_df, bads_df : pd.DataFrame, pd.DataFrame
    """
    assert isinstance(eeg_df, pd.DataFrame)
    assert feats in ['power', 'coherence']

    # compute features
    feat_lst = []  # collect feature arrays for each window
    fnames = None  # save feature names

    n_idx_slide = (WINDOW_LENGTH - WINDOW_OVERLAP) * SR
    #! TODO: np.floor
    start_idxs = range(0, eeg_df.shape[0] - WINDOW_LENGTH, n_idx_slide)
    end_idxs = [si + (WINDOW_LENGTH * SR) for si in start_idxs]

    i=0
    for si, ei in tqdm(zip(start_idxs, end_idxs), total = len(start_idxs)):
        win_df = eeg_df.iloc[si:ei]

        if do_interp:
            # for neurablescreen
            interpolation_kwargs=dict(
                thresh_reject=thresh_reject,    # Used if rejectmode is 'Vpeak2peak_manual'
                thresh_flat=0,                  # Used if rejectmode is 'Vpeak2peak_manual'
                n_interpolate=7,                # Max number of channels to interpolate before file considered bad
            )
            bads_array = np.zeros([len(start_idxs), eeg_df.shape[1]])   # pre-allocate
            win_df, bads_mask = neurablescreen_wrapper(win_df, interpolation_kwargs)
            bads_array[i,:] = bads_mask
        i += 1

        sdb, nms = extract_power_band_metrics(
            win_df,
            eeg_metric = feats,
            channels = win_df.columns,
            s_rate = 500,
            bands = BANDS,
        )
        
        feat_lst.append(sdb)
        #! pain to avoid None return screwup on the last window
        if nms is not None:
            fnames = nms

    # extract corresponding time points
    start_ts = eeg_df.index.values[start_idxs]
    assert len(feat_lst) == len(start_ts)

    # handle case where window <1s so no features
    # do this after the loop completes to apply corresponding trimming to time indices
    has_feats = [f is not None for f in feat_lst]
    feat_lst = [f for f, b in zip(feat_lst, has_feats) if b]
    start_ts = [t for t, b in zip(start_ts, has_feats) if b]

    # enframe features
    feat_np = np.vstack(feat_lst)
    feat_df = pd.DataFrame(data = feat_np, columns=fnames, index=start_ts)

    if do_interp:
        # enframe bad channel labels matrix
        bads_array = bads_array[np.array(has_feats),:]
        bads_df = pd.DataFrame(data = bads_array, columns = eeg_df.columns, index=start_ts)
        return feat_df, bads_df

    return feat_df


def extract_power_band_metrics(data_segment, **params):
    """
    From a given EEG data segment, extract EEG features
    :param data_segment: DataFrame, table of electrode timeseries segment data
    :param eeg_metric: str, 'coherence' or 'power', for spectral coherence, or power band calculations
    :param channels: list, channel names to use in the feature calculations
    :param s_rate: float, data sampling rate
    :param bands: dict, dictionary of tuple of start and stop frequencies for desired power bands
    :return: (ndarray, list), calculated user features and corresponding names; or None if window was too small
    """
    eeg_metric = params['eeg_metric']
    _avail_metrics = ['power', 'asymmetry', 'coherence']
    assert eeg_metric in _avail_metrics, f"Supported EEG metrics: {', '.join(_avail_metrics)}"

    channels = params['channels']
    srate = params['s_rate']
    data = data_segment.loc[:, channels].values
    bands = params['bands']
    band_names = list(bands.keys())
    band_names.sort()
    feature_names = []
    metrics = []

    freq = None
    pwr = None

    freq_mins = [band[0] for band in bands.values()]
    freq_maxs = [band[1] for band in bands.values()]

    # Enforce minimum of 1 seconds for data
    if len(data) / srate < 1.0:
        log = logging.getLogger(__name__)
        log.warning("Extract Power Metrics: At least 1 second of data required")
        return None, None

    # Multitaper power band calculation
    if eeg_metric == 'power':
        for freq_min, freq_max in zip(freq_mins, freq_maxs):
            pwr, freq = psd_array_multitaper(data.T, srate, adaptive=True, normalization='full', verbose=False,
                                             fmin=freq_min, fmax=freq_max)
            metrics.append(np.log10(pwr.sum(axis=1)))
        for band in band_names:
            for chan in channels:
                feature_names.append("{:s}_{:s}".format(chan, band))

    elif eeg_metric == 'asymmetry':
        n_chan = data.shape[1]//2
        data_left = data[:, :n_chan]
        data_right = data[:, n_chan:]
        for freq_min, freq_max in zip(freq_mins, freq_maxs):
            pwr_left, freq = psd_array_multitaper(data_left.T, srate, adaptive=True, normalization='full',
                                                  verbose=False, fmin=freq_min, fmax=freq_max)
            pwr_right, freq = psd_array_multitaper(data_right.T, srate, adaptive=True, normalization='full',
                                                  verbose=False, fmin=freq_min, fmax=freq_max)
            metrics.append(np.log10(pwr_right.sum(axis=1)) - np.log10(pwr_left.sum(axis=1)))
        for band in band_names:
            for chan_left, chan_right in zip(channels[:n_chan], channels[n_chan:]):
                feature_names.append("{:s}_{:s}_{:s}".format(chan_left, chan_right, band))


    # Multitaper coherence calculations
    elif eeg_metric == 'coherence':
        c_idxs = np.stack([c_pair for c_pair in combinations(range(len(channels)), 2)])
        coh = spectral_connectivity([data.T], sfreq=srate, fmin=freq_mins, fmax=freq_maxs, faverage=True,
                                    method='coh', mode='multitaper', verbose=False)
        pwr, freq = coh[:2]
        for band in band_names:
            for c_pair in c_idxs:
                c0, c1 = c_pair
                feature_names.append("{:s}_{:s}_{:s}".format(channels[c0], channels[c1], band))

        for _, band in enumerate(band_names):
            # Flatten coherence output container
                metrics = pwr.T[pwr.T > 0].reshape(-1)
                #coh_band = pwr[:, :, idx].T
                #metrics.append(coh_band[coh_band != 0].reshape(-1))

    return np.hstack(metrics), feature_names



if __name__ == "__main__":
    import os
    DIR_IN = "/mnt/data/stroop2020/preproc/"

    # input interface
    assert(path.isdir(DIR_IN))
    rec_fns = os.listdir(DIR_IN)
    fps_in = [path.join(DIR_IN, fn) for fn in rec_fns]

    FP_PROTO = fps_in[0]
    csv_data = pd.read_csv(FP_PROTO, delimiter=",", header=0,
                            index_col=0, low_memory=False)
    eeg_df = csv_data.iloc[:, 1:21]

    # Sliding features
    print(f"Computing power features for {FP_PROTO}")
    pow_df, bads_df = eeg_features(eeg_df)
    print(pow_df.head())

    print(f"Computing coherence features for {FP_PROTO}")
    coh_df, bads_df = eeg_features(eeg_df, feats="coherence")
    print(coh_df.head())

