# zunächst importieren wir das MNE Package & weitere Packages, damit wir sie verwenden können
import logging
logging.getLogger('numexpr').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg is non-interactive.*')
import mne 
from mne import viz
from mne.io import RawArray
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.find_noisy_channels import find_bad_by_ransac
import matplotlib.pyplot as plt
import numpy as np
from meegkit.detrend import detrend
from meegkit.dss import dss_line
import pandas as pd
from helper_functions import *

def quick_prep(raw, bad_channels_passive, bad_channels_active):
    # Standard Map laden und an Datensatz zufügen
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)


    # Elektrodennummerierunge
    channels = pd.read_csv('electrode_info.csv')
    channels = channels[1:25].reset_index()['electrodes']
    channel_names = pd.DataFrame(channels, index=range(24))
    channel_names['ID'] = range(24)

    resample_freq = 200
    _ = raw.resample(resample_freq)

    raw_passive, raw_active = split_dataset(raw)

    # Verwende die Funktion "detrend_baselines()" um die Aktivität um die Baseline um die 0 mV zu bringen
    detrend_baselines(raw_passive)
    detrend_baselines(raw_active)

    # remove channels
    raw_passive_elect = remove_bad_channels(bad_channels_passive, raw_passive)
    raw_active_elect = remove_bad_channels(bad_channels_active, raw_active)

    # Verwende "dss_line()" für das "passive" Datenset ...
    raw_passive_filtered = filter_line_power(raw_passive_elect, line_freq=50, filter_stärke=5)

    # ... und für das "aktive"
    raw_active_filtered = filter_line_power(raw_active_elect, line_freq=50, filter_stärke=5)

    # Die Ausführung dieser Funktion kann etwas dauern 
    raw_passive_cleaned = apply_ica(raw_passive_filtered)
    raw_active_cleaned = apply_ica(raw_active_filtered)

    return raw_passive_cleaned, raw_active_cleaned