
# %% IMPORTS
import numpy as np 
import itertools
import time
import eego_sdk
import os 
import mne
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %%

# FUNCTIONS


def amplifier_to_id(amplifier):
  return '{}-{:06d}-{}'.format(amplifier.getType(), amplifier.getFirmwareVersion(), amplifier.getSerialNumber())

def test_impedance(amplifier):
  stream = amplifier.OpenImpedanceStream()

  print('stream:')
  print('  channels.... {}'.format(stream.getChannelList()))
  print('  impedances.. {}'.format(list(stream.getData())))

def create_mne_info(amplifier):
    # You'll need to define channel names, types, and sampling frequency
    channels = amplifier.getChannelList()
    channel_info = pd.read_csv('electrode_info.csv') # load csv file with info create_electrode_info()
    ch_names =  list(channel_info['electrodes'].loc[1:24]) # List of channel names
    ch_types = ['eeg' for i in range(24)]  # List of channel types, e.g., 'eeg'
    sfreq = amplifier.getSamplingRatesAvailable()[0]      # Sampling frequency

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return info

def save_to_fif(amplifier, data, filename):
    info = create_mne_info(amplifier)
    
    # Assuming 'data' is a NumPy array with shape (n_channels, n_samples)
    raw = mne.io.RawArray(data, info)

    # Save to a .fif file
    raw.save(filename, overwrite=True)

def stream_signal(amplifier):
    rates = amplifier.getSamplingRatesAvailable()
    ref_ranges = amplifier.getReferenceRangesAvailable()
    bip_ranges = amplifier.getBipolarRangesAvailable()
    rate = rates[0]
    stream = amplifier.OpenEegStream(rate, ref_ranges[0], bip_ranges[0])

    print('stream:')
    print('  rate:       {}'.format(rate))
    print('  channels:   {}'.format(stream.getChannelList()))


    with open('%s-eeg.txt' % (amplifier_to_id(amplifier)), 'w') as eeg_file:
        # get data for 10 seconds, 0.25 seconds in between
        t0 = time.time()
        t1 = t0 + duration
        interval = 1
        tnext = t0

        while time.time() < t1:
            tnext = tnext + interval
            delay = tnext - time.time()
            if delay > 0:
                time.sleep(delay)

            try:
                data = stream.getData()
                print(type(data))
                print('  [{:04.4f}] delay={:03} buffer, channels: {:03} samples: {:03}'.format(time.time() - t0, delay, data.getChannelCount(), data.getSampleCount()))
                for s in range(data.getSampleCount()):
                    for c in range(data.getChannelCount()):
                        eeg_file.write(' %f' % (data.getSample(c, s)))
                    eeg_file.write('\n')
            except Exception as e:
                print('error: {}'.format(e))

def create_electrode_info():
   # Data to be transformed into a dictionary
    manual_data = """
    NA GND 34 68
    1 Fp1 33 67
    2 Fp2 32 66
    3 F9 31 65
    4 F7 30 64
    5 F3 29 63
    6 Fz 28 62
    7 F4 27 61
    8 F8 26 60
    9 F10 25 59
    10 T9 24 58
    11 T7 23 57
    12 C3 22 56
    13 C4 21 55
    14 T8 20 54
    15 T10 19 53
    16 P9 18 52
    17 P7 17 51
    18 P3 16 50
    19 Pz 15 49
    20 P4 14 48
    21 P8 13 47
    22 P10 12 46
    23 O1 11 45
    24 O2 10 44
    NA REF 1 35
    """

    # Splitting the data into lines and then into columns
    rows = manual_data.strip().split("\n")
    parsed_data = [row.split() for row in rows]

    # Creating a list of dictionaries
    electrode_data = []
    for row in parsed_data:
        channel_number, electrode, active, shield = row
        electrode_data.append({
            "channel_number": channel_number,
            "electrodes": electrode,
            "active": active,
            "shield": shield
        })

    # Display the first 5 entries for review
    #print(electrode_data[:5])
    electrode_df = pd.DataFrame(electrode_data)
    electrode_df.to_csv('electrode_info.csv')

    return electrode_df

def init():
    for line in lines:
        line.set_data([],[])
    ax.set_ylim(-1, 1) 
    
    return lines

# Plotting functions
def update_plot(frame, stream, n_channels):
    global ydata

    data = stream.getData()
    data_all = []

    for s in range(data.getSampleCount()):
        data_samples = []
        for c in range(n_channels):
            data_samples.append(data.getSample(c, s))
            eeg_file.write(' %f' % (data.getSample(c, s)))
        
        data_all.append(data_samples)
        eeg_file.write('\n')

    data_all = np.array(data_all)
    data_all_channels = data_all.T
    current_time = time.time() - t0
    window_size = 1  # Window size in seconds (adjust as needed)
    start_time = max(0, current_time - window_size/2)
    end_time = current_time + window_size/2
    dy = (data_all_channels.min() - data_all_channels.max()) * 0.3

    for i in range(5):
        new_ydata = data_all_channels[i] + i * dy # np.mean(data_all_channels, 0)

        if len(ydata[i]) > window_size * sfreq:
            ydata[i] = ydata[i][-int(window_size * sfreq):]

        if isinstance(new_ydata, np.float64):
            ydata[i].append(new_ydata)
        else:
            ydata[i].extend(list(new_ydata))
            
        xdata = np.linspace(current_time - window_size, current_time, len(ydata[i]))
        lines[i].set_data(xdata, ydata[i])  # Update the line plot with new data
    
    ax.set_xlim(start_time, end_time)
        
    return lines

def plot_realtime(stream, n_channels):
    global lines
    lines = [ax.plot([], [])[0] for _ in range(5)]  # Create a line for each of the first five channels
    ani = FuncAnimation(fig, update_plot, fargs=(stream, n_channels), frames=range(duration * sfreq),
                    init_func=init, blit=True, interval=2000/sfreq)
    plt.show()

def init_imp():
    bar_container = ax.bar([], [], lw=10, alpha=0.5)


def update_impedance_plot(frame, stream, channel_names, bar_container):
    impedances = list(stream.getData())
    for bar, new_height in zip(bar_container, impedances):
        bar.set_height(new_height)

    ax.set_ylim(0, np.max(impedances)+1000)  # Adjust the upper limit as needed

    # Return the bars (artist objects) as a tuple
    return bar_container.patches

def plot_impedances(stream, channel_names):
    fig, ax = plt.subplots()
    #ax.set_ylim(top=55)  # set safe limit to ensure that all data is visible.
    #data = list(stream.getData())
    ax.set_ylim(0, 8000000)  # Adjust the upper limit as needed

    bar_container = ax.bar(channel_names, np.array(1)*26, lw=10, alpha=0.5)
    ani = FuncAnimation(fig, update_impedance_plot, fargs=(stream, channel_names, bar_container), frames=range(duration * sfreq),
                              repeat=False, blit=True)
    plt.show()


# %% 
# MAIN

if __name__ == '__main__':
    # prepare
    factory = eego_sdk.factory()
    v = factory.getVersion()
    #print('version: {}.{}.{}.{}'.format(v.major, v.minor, v.micro, v.build))

    print('delaying to allow slow devices to attach...')
    time.sleep(1)

    # STREAM
    amplifiers = factory.getAmplifiers()
    amplifier = amplifiers[0]
    channels = amplifier.getChannelList()
    rates = amplifier.getSamplingRatesAvailable()
    ref_ranges = amplifier.getReferenceRangesAvailable()
    bip_ranges = amplifier.getBipolarRangesAvailable()
    sfreq = rates[0]
    channel_info = pd.read_csv('electrode_info.csv')
    channel_names =  list(channel_info['electrodes'].loc[1:24])
    channel_names.append('REF')
    channel_names.append('GND')

    # test impedances
    test_imp = True
    if test_imp:
        stream = amplifier.OpenImpedanceStream()    
        #print('  impedances.. {}'.format(list(stream.getData())))
    else:
        stream = amplifier.OpenEegStream(sfreq, ref_ranges[0], bip_ranges[0])

    # setup for the plotting
    duration = 1
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    n_channels = 24
    # Number of channels you want to plot
    n_channels_to_plot = 5
    # Initialize ydata as a list of empty lists, one for each channel
    ydata = [[] for _ in range(n_channels_to_plot)]
    xdata = []

    # READOUT 
    with open('%s-eeg.txt' % (amplifier_to_id(amplifier)), 'w') as eeg_file:

        t0 = time.time()
        t1 = t0 + duration
        interval = 0.25
        impedance_interval = 0.5
        current_imp = time.time()
        tnext = t0

        while time.time() < t1:
            tnext = tnext + interval
            delay = tnext - time.time()
            if delay > 0:
                time.sleep(delay)

            if test_imp:
                plot_impedances(stream, channel_names)
            else:
                plot_realtime(stream, n_channels, channel_names)

            if time.time() - current_imp > impedance_interval:
                test_impedance(amplifier)
                current_imp = time.time()
            

            # data = stream.getData()
            # data_samples = []
            # array_data = []
            # print('  [{:04.4f}] delay={:03} buffer, channels: {:03} samples: {:03}'.format(time.time() - t0, delay, data.getChannelCount(), data.getSampleCount()))
            # for s in range(data.getSampleCount()):
            #     for c in range(24):
            #         #data_samples.append(data.getSample(c, s))
            #         eeg_file.write(' %f' % (data.getSample(c, s)))
            #     eeg_file.write('\n')
            #     #array_data.append(data_samples)

            

    #array_data = np.array(array_data)
    #array_data = np.loadtxt('%s-eeg.txt' % (amplifier_to_id(amplifier)))
    #print('array', array_data.shape)
    #save_to_fif(amplifier, np.array(array_data), 'output_file.fif')
    
    #save_to_fif(amplifier, array_data.T, 'output_file_TEST1.fif')

    print('DONE')
# %%
