#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@time    :   2022/02/10 13:31:09
@author  :   rosagross
@contact :   grossmann.rc@gmail.com
'''

import numpy as np
import os
import re
import csv
import random
import yaml
import eego_sdk
import mne
import pandas as pd
import time
import matplotlib.pyplot as plt
from datetime import datetime
from psychopy import core
from psychopy.visual import Window, TextStim, Circle, Polygon, ShapeStim
from psychopy.event import waitKeys, Mouse
from psychopy.hardware import keyboard
import random

class ExperiSession():
    def __init__(self, output_str, output_dir, settings_file, subject_ID, session_type):
        """
        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs)
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        subject_ID : int
            ID of the current participant
        """

        self.output_str = output_str
        self.output_dir = output_dir
        self.settings = settings_file
        self.subject_ID = subject_ID
        self.session_type = session_type

        # Read YAML file
        with open(settings_file, 'r') as stream:
            self.settings = yaml.safe_load(stream)
        
        self.exp_duration = self.settings['Task settings']['Experiment duration']
        self.stim_duration = self.settings['Task settings']['Stimulus duration']
        self.task_duration = self.stim_duration * 5
        print("Stimulation duration", self.stim_duration)
        self.win = Window(**self.settings['window'])

        # set if this is a "active" session or a "passive" session
        if session_type == 1:
            self.session_mode = "active"
            self.blue_list = []
            self.red_list = []
            self.yellow_list = []
            # create the stimuli 
            print("active SESSION")
            self.color_str, self.number_list, self.color_list = self.create_math_stim()
            
            # store the sums in a dictionary
            self.sums = {colour : 0 for colour in self.color_str}
            print(self.sums)
        else:
            self.session_mode = "passive"
            print("passive SESSION")

        # initialize the keyboard for the button presses
        self.kb = keyboard.Keyboard()


    def amplifier_to_id(self, amplifier):
        return '{}-{:06d}-{}'.format(amplifier.getType(), amplifier.getFirmwareVersion(), amplifier.getSerialNumber())
    

    def display_text(self, text, keys=None, duration=None, **kwargs):
        """ Displays text on the window and waits for a key response.
        The 'keys' and 'duration' arguments are mutually exclusive.

        parameters
        ----------
        text : str
            Text to display
        keys : str or list[str]
            String (or list of strings) of keyname(s) to wait for
        kwargs : key-word args
            Any (set of) parameter(s) passed to TextStim
        """
        if keys is None and duration is None:
            raise ValueError("Please set either 'keys' or 'duration'!")

        if keys is not None and duration is not None:
            raise ValueError("Cannot set both 'keys' and 'duration'!")

        stim = TextStim(self.win, text=text, **kwargs)
        stim.draw()
        self.win.flip()

        if keys is not None:
            waitKeys(keyList=keys)
        
        if duration is not None:
            core.wait(duration)
    

    def create_math_stim(self):
        # make a list with all coloured circles that will occur 
        nr_circles = int(np.ceil(self.exp_duration / self.stim_duration))
        color_list = [np.random.randint(0,3) for _ in range(nr_circles)]
        random.shuffle(color_list)
        color_str = ["red", "green", "blue"]
        circle_stim_list = []

        # make a list with all numbers that will occur

        number_list = [np.random.randint(1,10)]
        
        for _ in range(nr_circles):
            next_num = np.random.randint(1,10)
            while next_num == number_list[-1]:
                next_num = np.random.randint(1,10)
            number_list.append(next_num)

        number_list[-1] = 0
        number_stim_list = []


        
        # coloured circle with a number and safe it to the corresponding color list
        for id_, number in zip(color_list, number_list):
            #print('hi', color_str[id_], number)
            circle_stim_list.append(Circle(self.win, fillColor=color_str[id_]))
            number_stim_list.append(TextStim(self.win, height=1, text=number))

        return color_str, number_list, color_list
    
    def create_mne_info(self, amplifier):
        # You'll need to define channel names, types, and sampling frequency
        channel_info = pd.read_csv('electrode_info.csv') # load csv file with info create_electrode_info()
        ch_names =  list(channel_info['electrodes'].loc[1:24]) # List of channel names
        ch_types = ['eeg' for i in range(24)]  # List of channel types, e.g., 'eeg'
        sfreq = amplifier.getSamplingRatesAvailable()[0]      # Sampling frequency

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        return info
    
    
    def save_to_fif(self, amplifier, data, filename):
        info = self.create_mne_info(amplifier)
        
        # Assuming 'data' is a NumPy array with shape (n_channels, n_samples)
        raw = mne.io.RawArray(data, info)

        # Save to a .fif file
        raw.save(filename, overwrite=True)

    def save_session(self, amplifier):
        # save if its passive or active session

        array_data = np.loadtxt(f'{self.output_dir}\sub-{self.subject_ID}_ses-{self.session_mode}.txt')
        print('array', array_data.shape)
        channels_to_save = 24
        channel_data = array_data.T[:24]
        self.save_to_fif(amplifier, channel_data, f'{self.output_dir}\sub-{self.subject_ID}_ses-{self.session_mode}.fif')
        

    def save_sample(self, stream, channels, eeg_file):
        
        # write the recording for one interval (e.g. 0.25 sec) to a file 
        data = stream.getData()
        data_all = []

        for s in range(data.getSampleCount()):
            data_samples = []
            for c in range(len(channels)):
                data_samples.append(data.getSample(c, s))
                eeg_file.write(' %f' % (data.getSample(c, s)))
            
            data_all.append(data_samples)
            eeg_file.write('\n')


    def test_impedance(self, amplifier):
        stream = amplifier.OpenImpedanceStream()

        print('stream:')
        print('  channels.... {}'.format(stream.getChannelList()))
        print('  impedances.. {}'.format(list(stream.getData())))

    def run(self):

        #factory = eego_sdk.factory()
        #v = factory.getVersion()
        #print('version: {}.{}.{}.{}'.format(v.major, v.minor, v.micro, v.build))

        #print('delaying to allow slow devices to attach...')
        #time.sleep(1)

        # STREAM
        #amplifiers = factory.getAmplifiers()
        #amplifier = amplifiers[0]
        #channels = amplifier.getChannelList()
        #rates = amplifier.getSamplingRatesAvailable()
        #ref_ranges = amplifier.getReferenceRangesAvailable()
        #bip_ranges = amplifier.getBipolarRangesAvailable()
        #sfreq = rates[0]
        #stream = amplifier.OpenEegStream(sfreq, ref_ranges[0], bip_ranges[0])
        
        #self.test_impedance(amplifier)
        # setup for the plotting
        # fig, ax = plt.subplots()
        # line, = ax.plot([], [])
        # n_channels = 24
        # Number of channels you want to plot
        #n_channels_to_plot = 5

        # Initialize ydata as a list of empty lists, one for each channel
        # ydata = [[] for _ in range(n_channels_to_plot)]
        # xdata = []

        # show window and wait for "OKAY" from experimenter
        self.display_text("Warte bis der Experimenter bereit ist", "space")

        # READOUT 
        with open(f'{self.output_dir}\sub-{self.subject_ID}_ses-Solution.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Lösung', 'Farbe'])

            t0 = time.time()
            t1 = t0 + self.exp_duration
            interval = 0.25
            tnext = t0
            t_trial = t0
            trial_count = 0

            fix_dot = ShapeStim(self.win, units='pix', fillColor='black', vertices='cross', lineWidth=1, size=(50,50))
            fix_dot.draw()
            self.win.flip()

            while time.time() < t1:
                

                if (time.time() - t_trial) > self.stim_duration: 
                    # reset trial timer
                    t_trial = time.time()


                    if self.session_mode == "active":
                        if (trial_count % 5 == 0) and (trial_count > 0):
                            colour_task = random.randint(0,2)
                            if colour_task == 0:
                                self.display_text(f"Welche Summe hat die Farbe \n Rot", duration=4)
                                writer.writerow([self.sums['red'], 'ROT'])
                            
                            elif colour_task == 1:    
                                self.display_text(f"Welche Summe hat die Farbe \n Grün", duration=4)
                                writer.writerow([self.sums['green'], 'GRÜN'])

                            elif colour_task == 2:    
                                self.display_text(f"Welche Summe hat die Farbe \n Blau", duration=4)
                                writer.writerow([self.sums['blue'], 'BLAU'])

                            # reset trial timer
                            t_trial = time.time()
                            # reset sum
                            self.sums = {colour : 0 for colour in self.color_str}

                        # put next stimulus
                        number = self.number_list[trial_count]
                        color = self.color_str[self.color_list[trial_count]]
                        Circle(self.win, fillColor=color).draw()
                        TextStim(self.win, height=1, text=number).draw()
                        self.sums[color] += number
                        
                        self.win.flip()
                        trial_count += 1
                
                tnext = tnext + interval
                delay = tnext - time.time()
                
                if delay > 0:
                    time.sleep(delay)

                # self.save_sample(stream, channels, eeg_file)

        # self.save_session(amplifier)

        self.display_text("WELL DONE!", "space")
        if self.session_mode == "active":
            print(self.sums)
            #self.display_text(f"Korrekte Summe:\n Rot:{self.sums['red']}, Grün:{self.sums['green']}, Blau:{self.sums['blue']}", "space")

            

                




