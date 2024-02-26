# Hilfreiche Funktionen
# zunächst importieren wir das MNE Package, damit wir es verwenden können
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


def plot_signal(raw, start=0, end=None, electrodes=range(24)):
    """ 
    Die Funktion erstellt eine Abbildung der gemessenen Daten, für eine ausgwählte Zeit und für ausgewählte Funktionen.
    @param raw: Datensatz
    @param start: Signal Start (in Sekunden)
    @param end: Signal Ende (in Sekunden)
    """
    sfreq = int(raw.info['sfreq'])
    start = start*sfreq
    if not end:
        end = raw.get_data().shape[1]
    else: 
        end = end*sfreq
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(raw.times[start:end], raw.get_data()[electrodes, start:end].T*1e6, linewidth=0.4)
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [muV]')
    plt.title('Activity')
    plt.show()

def threshold_electrode(raw, threshold):
    '''
    threshold: value in µV
    '''

    # load electrode names 
    electrode_names = np.asarray(raw.info['ch_names'])
    
    # Find the electrode(s) over threshold
    electrodes_above_threshold = np.any(raw.get_data() > threshold*1e-6, axis=1)
    
    # Print the index(es) of the electrode(s)
    electrode_indices = np.where(electrodes_above_threshold)[0]
    print(f"Electrode(s) with values above {threshold}: Name {np.array(electrode_names[electrode_indices])}")
    
def detrend_baselines(raw):
    X = raw.get_data().T # transpose so the data is organized time-by-channels
    X, _, _ = detrend(X, order=1)
    X, _, _ = detrend(X, order=6)
    raw._data = X.T  # overwrite raw data
    raw.set_eeg_reference('average', projection=True)  #compute the reference

def detect_bad_channels(raw, corr_thresh=0.75):
    bads, _ = find_bad_by_ransac(
         data = raw.get_data(),
         sample_rate = raw.info['sfreq'],
         complete_chn_labs = np.asarray(raw.info['ch_names']),
         chn_pos = np.stack([ch['loc'][0:3] for ch in raw.info['chs']]),
         exclude = [],
         corr_thresh = corr_thresh
         )    
    return bads

def remove_bad_channels(bads, raw):
    raw_copy = raw.copy()
    raw_copy.info['bads'] = bads
    raw_copy.interpolate_bads()
    raw_copy.set_eeg_reference('average', projection=True)  #compute the reference
    raw_copy.apply_proj()
    
    return raw_copy 

def compare_power_plot(raw_passive, raw_active, fmin, fmax): 

    sfreq = reconst_raw_passive.info['sfreq']
    psds_passive, freqs = mne.time_frequency.psd_array_welch(raw_passive.get_data(), sfreq=sfreq, fmin=fmin, fmax=fmax, average='mean')
    psds_active, freqs = mne.time_frequency.psd_array_welch(raw_active.get_data(), sfreq=sfreq, fmin=fmin, fmax=fmax, average='mean')

    # Umrechnen in bezibel (µV²/Hz)
    db_psds_passive = 10 * np.log10(psds_passive / (1e-6)**2)
    db_psds_active = 10 * np.log10(psds_active / (1e-6)**2)

    fig,(ax1,ax2) = plt.subplots(ncols=2)

    im,cm = viz.plot_topomap(np.mean(db_psds_passive,1), reconst_raw_passive.info, axes=ax1, show=False)
    im,cm = viz.plot_topomap(np.mean(db_psds_active,1), reconst_raw_active.info, axes=ax2,show=False)
    ax1.set_title('Passive')
    ax2.set_title('Active')

    # Position der colorbar
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9

    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title("Energy",fontsize=8) # title on top of colorbar


def split_dataset(raw):
    # lese die event markierungen aus dem Datenset heraus
    events, description = mne.events_from_annotations(raw)

    # get only the "new segment" events
    segment_events = events[events[:,2] == description['New Segment/']]
    raw_passive = raw.copy().crop(tmin=segment_events[0,0] / raw.info['sfreq'], 
                                  tmax=segment_events[2,0] / raw.info['sfreq'])
    raw_active = raw.copy().crop(tmin=segment_events[-1,0] / raw.info['sfreq'])
    return raw_passive, raw_active