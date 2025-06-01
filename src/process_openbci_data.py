import mne
import json

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import mne
from mne.channels import read_layout

DATA_DIR = "/home/reuben/Documents/eeg-data/"

def load_openbci_data(jsons_path, prompt_type, verbose=False):
    
    samples_path = jsons_path + "/eeg_samples-" + prompt_type + ".json"
    markers_path = jsons_path + "/eeg_markers-" + prompt_type + ".json"

    with open(samples_path, 'r') as json_file1:
        eeg_samples = json.load(json_file1)

    with open(markers_path, 'r') as json_file2:
        eeg_markers = json.load(json_file2)

    buffered_sample_number = len(eeg_samples)
    channel_number = len(eeg_samples[0])

    if verbose:
        print("OpenBCI Dataset", name, "Loaded")
        print("Dataset contains", buffered_sample_number, "buffered samples, and", channel_number, "channels")
        print("----------Printing Size of Each Channel Buffer---------")
        for i in range(buffered_sample_number):
            print()
            for j in range(channel_number):
                print(len(eeg_samples[i][j]), end=" ")
            print("| marker =", eeg_markers[i], end="")
        print("\n-----------------------------------------------------")

    eeg_samples_unbuffered = []
    eeg_markers_unbuffered = []

    for channel in range(channel_number + 1):
        eeg_samples_unbuffered.append([])

    prev_marker = 0
    for i, buffer in enumerate(eeg_samples):
        if i <= 1:
            continue
        new_marker = eeg_markers[i]
        buffer_size = len(buffer[0])
        for b in range(buffer_size):
            # sample = []
            for channel in range(channel_number):
                # sample.append(buffer[channel][b])
                scaled_data_point = buffer[channel][b] / 1000000 # BrainFlow returns uV, convert to V for MNE
                eeg_samples_unbuffered[channel].append(scaled_data_point)

            eeg_markers_unbuffered.append(new_marker)
            if (b==0) and new_marker != prev_marker:
                eeg_samples_unbuffered[channel_number].append(new_marker)
                prev_marker = new_marker
            else:
                eeg_samples_unbuffered[channel_number].append(0)
    
    # print(len(eeg_samples_unbuffered), len(eeg_samples_unbuffered[0]))
    # print(eeg_samples_unbuffered)

    return eeg_samples_unbuffered, eeg_markers_unbuffered

def convert_to_mne(name, save_name, save_path, samples, markers, save=True):
    # adapted from: https://brainflow.readthedocs.io/en/stable/notebooks/brainflow_mne.html 

    # board_id = BoardIds.SYNTHETIC_BOARD
    board_id = BoardIds.CYTON_BOARD
    
    params = BrainFlowInputParams()

    eeg_channels = BoardShim.get_eeg_channels(board_id.value)
    eeg_data = np.array(samples)#[eeg_channels, :]
    # eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE

    print("eeg_channels", eeg_channels)
    print("len(eeg_data)", len(eeg_data))
    print("len(eeg_data[0])", len(eeg_data[0]))
    print("eeg_data[0]", eeg_data[0])
    # print("eeg_data", eeg_data)

    # Creating MNE objects from brainflow data arrays
    ch_types = ['eeg'] * len(eeg_channels)
    ch_types.append('stim')
    ch_names = BoardShim.get_eeg_names(board_id.value)
    ch_names.append('STI 014')


    print("ch_types", ch_types)
    print("ch_names", ch_names)
    
    sfreq = BoardShim.get_sampling_rate(board_id.value)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(eeg_data, info)
    print("----------events--------")
    events = mne.find_events(raw)#, stim_channel="STI 014")
    print(events[:5])  # show the first 5 events
    raw.set_montage("standard_1005")
    raw = raw.filter(l_freq=0.2, h_freq=40)


    # plot data
    raw.plot()
    raw.plot_psd(average=False)
    plt.show()

    # Create EPOCHS
    event_dict = {
        "Rest": 1,
        "Left Fist": 2,
        "Right Fist": 3
    }
    tmin, tmax = -0.5, 2  # define epochs around events (in s)
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, preload=True)

    print(epochs)
    print(epochs.event_id)

    # epochs["Left Hand"].average().plot()
    # epochs["Right Hand"].average().plot()
    # epochs["Rest"].average().plot()

    # # Plot topomaps
    # times = np.arange(0, 1, 0.1)
    # epochs.average().plot_topomap(times, ch_type='eeg')
    # epochs["Left Hand"].average().plot_topomap(times, ch_type='eeg')
    # epochs["Right Hand"].average().plot_topomap(times, ch_type='eeg')
    # epochs["Rest"].average().plot_topomap(times, ch_type='eeg')

    # Plot epochs
    epochs.plot(scalings='auto', events=True)

    plt.show()

    if save:
        # Save Epochs
        epochs.save(save_path + "/" + save_name + '-epo.fif', overwrite=True)


if __name__ == "__main__":
    prompt_type = 'MI'
    name = "data-reuben-2122-2205-3-classes"
    jsons_path = DATA_DIR + "/reuben-openbci/" + name
    samples, markers = load_openbci_data(jsons_path, prompt_type, verbose=True)
    convert_to_mne(name, name + '-' + prompt_type, jsons_path, samples, markers, save=False)