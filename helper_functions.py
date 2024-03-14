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
import contextlib
import io
import sys

@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

def plot_signal(raw, start=0, end=None, electrodes=range(24)):
    """ 
    Die Funktion erstellt eine Abbildung der gemessenen Daten, für eine ausgwählte Zeit und für ausgewählte Funktionen.
    @param raw: Datensatz
    @param start: Signal Start (in Sekunden)
    @param end: Signal Ende (in Sekunden)
    """

    # print which channels are plotted
    channels = pd.read_csv('electrode_info.csv')
    channels = channels[1:25].reset_index()['electrodes']
    channel_names = pd.DataFrame(channels, index=range(24))

    sfreq = int(raw.info['sfreq'])
    start = start*sfreq
    if not end:
        end = raw.get_data().shape[1]
    else: 
        end = end*sfreq

    fig = plt.figure(figsize=(12,5))
    if len(electrodes) > 10:
        plt.plot(raw.times[start:end], raw.get_data()[electrodes, start:end].T*1e6, linewidth=0.7)
    else: 
        plt.plot(raw.times[start:end], raw.get_data()[electrodes, start:end].T*1e6,
                  label=[str(channel_names['electrodes'][channel]) for channel in electrodes])
        leg = plt.legend(framealpha=0.5, fontsize=15, markerscale=3, loc='upper right')

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

def detect_bad_channels_ransac(raw, corr_thresh=0.75):
    bads, _ = find_bad_by_ransac(
         data = raw.get_data(),
         sample_rate = raw.info['sfreq'],
         complete_chn_labs = np.asarray(raw.info['ch_names']),
         chn_pos = np.stack([ch['loc'][0:3] for ch in raw.info['chs']]),
         exclude = [],
         corr_thresh = corr_thresh
         )    
    return bads

def detect_bad_channels(raw):
    '''
    Diese Funktion detektiert Kanäle mit schlechtem Signal.
    '''
    # load channel names
    channels = pd.read_csv('electrode_info.csv')
    channels = channels[1:25].reset_index()['electrodes']
    channel_names = pd.DataFrame(channels, index=range(24))

    bad_channels = []
    psds_overall, _ = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=200, fmin=70, fmax=100, average='mean')
    psds_mean = np.mean(psds_overall, 0)
    psds_std = np.std(psds_overall, 0)
    formula = np.mean(psds_mean) + 2*np.mean(psds_std)

    for i, channel in enumerate(raw.get_data()):
        channel_psd, _ = mne.time_frequency.psd_array_welch(channel, sfreq=200, fmin=70, fmax=100, average='mean')
        channel_psd_mean = np.mean(channel_psd)

        # check if the channels' mean psd is higher than the overal psd + 3*std
        if channel_psd_mean > formula:
            bad_channels.append(channel_names['electrodes'][i])
    
    print('bad channels:', bad_channels)

def remove_bad_channels(bads, raw):
    raw_copy = raw.copy()
    raw_copy.info['bads'] = bads
    raw_copy.interpolate_bads()
    raw_copy.set_eeg_reference('average', projection=True)  #compute the reference
    raw_copy.apply_proj()

    print('Kanäle erfolgreich entfernt!')
    
    return raw_copy 

def compare_power_topography(raw_passive, raw_active, frequency_band): 

    sfreq = raw_passive.info['sfreq']
    psds_passive, freqs = mne.time_frequency.psd_array_welch(raw_passive.get_data(), sfreq=sfreq, fmin=frequency_band[0], fmax=frequency_band[1], average='mean')
    psds_active, freqs = mne.time_frequency.psd_array_welch(raw_active.get_data(), sfreq=sfreq, fmin=frequency_band[0], fmax=frequency_band[1], average='mean')

    # Umrechnen in bezibel (µV²/Hz)
    db_psds_passive = 10 * np.log10(psds_passive /np.mean(psds_passive))
    db_psds_active = 10 * np.log10(psds_active /np.mean(psds_active))

    fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(10,5))

    #print('passive', np.mean(psds_passive, 1))
    #print('active', np.mean(psds_active, 1))

    im,cm = viz.plot_topomap(np.mean(db_psds_passive,1), raw_passive.info, axes=ax1, show=False)#, cmap='viridis')
    im,cm = viz.plot_topomap(np.mean(db_psds_active, 1), raw_active.info, axes=ax2, show=False)#, cmap='viridis')
    ax1.set_title('Passive')
    ax2.set_title('Active')

    # Position der colorbar
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9

    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.set_label('Power Density (μV**2/Hz)', fontsize=10, rotation=270, labelpad=15)

def split_dataset(raw):
    '''
    Teile das Datenset in passiv und aktiv. 
    '''

    # Use the context manager to suppress output
    with suppress_stdout():
        # lese die event markierungen aus dem Datenset heraus
        events, description = mne.events_from_annotations(raw)


    # get only the "new segment" events
    segment_events = events[events[:,2] == description['New Segment/']]
    raw_passive = raw.copy().crop(tmin=segment_events[0,0] / raw.info['sfreq'], 
                                  tmax=segment_events[1,0] / raw.info['sfreq'])
    raw_active = raw.copy().crop(tmin=segment_events[-1,0] / raw.info['sfreq'])

    # Lösche die ersten 5 sekunden, da die Teilnehmer in der Zeit ihren Blick auf das Fixierungs-Kreuz gerichtet haben
    start_time = 5
    _ = raw_passive.crop(tmin=start_time, tmax=end_time)

    print("Datenset erfolgreich geteilt!")
    return raw_passive, raw_active

def filter_line_power(raw, line_freq, filter_stärke=3):
    # Verwende "dss_line()" für das "passive" Datenset ...
     # Use the context manager to suppress output
    with suppress_stdout():
        X_zapline, noise = dss_line(raw.get_data().T, fline=line_freq, sfreq=raw.info['sfreq'], nremove=filter_stärke)
        raw_zapline = RawArray(X_zapline.T, raw.info)
    print(f'Filtern mit Stärke {filter_stärke} erfolgreich!')
    
    # additionally filter 0.5 Signal
    raw_filtered = raw_zapline.copy()
    _ = raw_filtered.filter(l_freq=1, h_freq=None)

    return raw_filtered

def get_ica_components(raw, dataset_type):
    '''
    Calculates ICA but does not reject components yet.
    '''

    # Verwende ICA - passive
    ica = ICA(n_components=12, max_iter="auto", random_state=97)
    _ = ica.fit(raw)
    
    # Calculate fraction of variance in EEG signal explained by first component
    #explained_var_ratio = ica.get_explained_variance_ratio(
    #raw, components=[3], ch_type="eeg"
    #)
    # This time, print as percentage.
    #ratio_percent = round(100 * explained_var_ratio["eeg"])
    #print(
    #    f"Fraction of variance in EEG signal explained by first component: "
    #    f"{ratio_percent}%"
    #)
    
    # get the labels
    ic_labels = label_components(raw, ica, method="iclabel")
    #print(ic_labels["labels"])
    
    # entferne nun alle ICA Komponenten, die nicht als "brain" oder "other" klassifiziert wurden. 
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]

    print(f'\nKomponenten im {dataset_type} Experiment:\n', labels)
    print('Komponenten IDs:\n', exclude_idx)

    return labels, ica

def delete_ica_components(raw, ica, labels, component_IDs):
    '''
    Rejects ICA components
    '''
    reconst_raw = raw.copy()
    _ = ica.apply(reconst_raw, exclude=component_IDs)
    
    print(f"Entfernen ICA Komponenten {np.array(labels)[component_IDs]} erfolgreich!")
    
    return reconst_raw


def apply_ica(raw):
    '''
    Diese Funktion führt die Independent Component Analyse aus.
    '''
    
    # Verwende ICA - passive
    ica = ICA(n_components=12, max_iter="auto", random_state=97)
    _ = ica.fit(raw)
    
    # Calculate fraction of variance in EEG signal explained by first component
    #explained_var_ratio = ica.get_explained_variance_ratio(
    #raw, components=[3], ch_type="eeg"
    #)
    # This time, print as percentage.
    #ratio_percent = round(100 * explained_var_ratio["eeg"])
    #print(
    #    f"Fraction of variance in EEG signal explained by first component: "
    #    f"{ratio_percent}%"
    #)
    
    # get the labels
    ic_labels = label_components(raw, ica, method="iclabel")
    #print(ic_labels["labels"])
    
    # entferne nun alle ICA Komponenten, die nicht als "brain" oder "other" klassifiziert wurden. 
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    
    reconst_raw = raw.copy()
    _ = ica.apply(reconst_raw, exclude=exclude_idx)
    
    print(f"Entfernen ICA Komponenten {np.array(labels)[exclude_idx]} erfolgreich!")
    
    return reconst_raw

def frequency_band_topography(raw, frequency_band):

    # Berechne das Power Spectrum und rechne es um in Dezibel (microV**2/Hz)
    psd, freq = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=200, average='mean')
    psd_db = psd.T #10*np.log10(psd.T/1e-12) 

    # Extract the relevant psd values
    freq_ids = np.where((freq > frequency_band[0]) & (freq <= frequency_band[1]))
    psd_band = np.mean(psd_db[freq_ids], 0)
    psd_band_integrated = psd_band**2

    fig, ax = plt.subplots(1, figsize=(8, 6))  # Adjust the figure size as needed
    im, _ = viz.plot_topomap(psd_band_integrated, raw.info, axes=ax, show=False, sensors=True)#, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Integrated Power', rotation=270, labelpad=15)
    plt.show()
    
def calculate_energy(raw, frequency_band):
    
    # Berechne das Power Spectrum und rechne es um in Dezibel (microV**2/Hz)
    psd, freq = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=200, average='mean')
    psd_log = np.log10(psd.T/np.mean(psd.T)) #10*np.log2(psd.T/1e-12) 
    psd_db = 10 * np.log10(psd.T/1e-12) 
    psd = psd.T

    # Extract the relevant psd values
    freq_ids = np.where((freq > frequency_band[0]) & (freq <= frequency_band[1]))
    psd_band = np.mean(psd[freq_ids], 0)
    
    # integrate power
    power = np.mean(psd_band)**2
    power_integrated = np.trapz(psd_band)
    print(power)
    print(psd_band)
    print(power_integrated)
    
    plt.plot(np.mean(psd, 1))
    
def compare_energy_plot(raw_passive, raw_active, frequency_band):
    
    # Berechne das Power Spectrum und rechne es um in Dezibel (microV**2/Hz)
    sfreq = raw_passive.info['sfreq']
    psds_passive, freqs = mne.time_frequency.psd_array_welch(raw_passive.get_data(), sfreq=sfreq, fmin=frequency_band[0], fmax=frequency_band[1], average='mean')
    psds_active, freqs = mne.time_frequency.psd_array_welch(raw_active.get_data(), sfreq=sfreq, fmin=frequency_band[0], fmax=frequency_band[1], average='mean')
    psds_passive = np.mean(psds_passive, 0)
    psds_active = np.mean(psds_active, 0)
    
    # integrate power
    power_passive = np.trapz(psds_passive)
    power_active = np.trapz(psds_active)
    print('Energie im passiven Zustand:', power_passive)
    print('Energie   im aktiven Zustand:', power_active)

    # Plot comparison 
    fig = plt.figure(figsize=(10,2))
    plt.plot(freqs, 10*np.log10(psds_active/np.mean(psds_passive)), label='active')
    plt.plot(freqs, 10*np.log10(psds_passive/np.mean(psds_active)), label='passive')
    plt.ylabel('µV**2/Hz (dB)')
    plt.xlabel('Frequenz (Hz)')
    #plt.yticks([-1, 0, 1])
    plt.legend(loc='upper right')
    plt.show()

def compare_amplitude_topography(raw_passive, raw_active, frequency_band):
    
    # filter data
    raw_passive_filtered = raw_passive.copy()
    _ = raw_passive_filtered.filter(l_freq=frequency_band[0], h_freq=frequency_band[1])
    raw_active_filtered = raw_active.copy()
    _ = raw_active_filtered.filter(l_freq=frequency_band[0], h_freq=frequency_band[1])

    # create topoplot
    fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(10,5))
    
    #print(np.mean(raw_passive_filtered,1))

    im,cm = viz.plot_topomap(np.mean(raw_passive_filtered.get_data(),1), raw_passive_filtered.info, axes=ax1, show=False)#, cmap='viridis')
    im,cm = viz.plot_topomap(np.mean(raw_active_filtered.get_data(),1), raw_active_filtered.info, axes=ax2, show=False)#, cmap='viridis')
    ax1.set_title('Passive')
    ax2.set_title('Active')

    # Position der colorbar
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9

    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.set_label('Amplitude (μV)', fontsize=10, rotation=270, labelpad=15)



