from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import mne
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/home/reuben/Documents/eeg-data/"

def process_physionet(subject, name, task, verbose=False, visualise=False, save=True):
    """
    Task 1 (open and close left or right fist)
    Task 2 (imagine opening and closing left or right fist)
    Task 3 (open and close both fists or both feet)
    Task 4 (imagine opening and closing both fists or both feet)
    """
    runs = [task + 2, task + 6, task + 10]

    #Get data and locate in to given path
    files = eegbci.load_data(subject, runs, path=DATA_DIR)
    #Read raw data files where each file contains a run
    raws = [read_raw_edf(f, preload=True) for f in files]

    # -------------------------------------
    if verbose:
        print("FILES:")
        for f in files:
            print(" - ", f)

        for raw in raws:
            print("RAW", raw.filenames)
            print(" - shape:", raw.get_data().shape)
    # -------------------------------------

    # Combine all loaded runs
    raw_obj = concatenate_raws(raws)
    raw_obj = raw_obj.filter(l_freq=0.2, h_freq=50)
    

    map = {
    'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2',
    'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1',
    'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6', 'Cp5.': 'CP5',
    'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4',
    'Cp6.': 'CP6', 'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2', 'Af7.': 'AF7',
    'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7',
    'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2',
    'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7', 'Ft8.': 'FT8',
    'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7',
    'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1',
    'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
    'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8',
    'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'}

    raw_obj.rename_channels(map)
    raw_obj.set_montage("standard_1005")

    print("Channels (before drop):", raw_obj.ch_names)
    channels_to_drop = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
                     'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
                     'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
                     'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 
                     'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 
                     'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 
                     'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 
                     'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 
                     'PO8', 'O1', 'Oz', 'O2', 'Iz']
    
    channels_to_keep = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "P7", "P8"]

    for keep in channels_to_keep:
        channels_to_drop.remove(keep)

    raw_obj.drop_channels(channels_to_drop)

    print("Channels (after drop):", raw_obj.ch_names)

    # Extract events from raw data
    events, event_ids = mne.events_from_annotations(raw_obj, event_id='auto')

    # Create EPOCHS
    tmin, tmax = -1, 4  # define epochs around events (in s)
    epochs = mne.Epochs(raw_obj, events, event_ids, tmin, tmax, baseline=None, preload=True)

    if save:
        # Save Epochs
        epochs.save(DATA_DIR + '/physionet-fifs-8-channel/' + name + '-epo.fif', overwrite=True)
    
    # -------------------------------------
    if verbose:
        # events = mne.events_from_annotations(raw_obj)
        print("events[1]:", events[1])
        print("Number of channels: ", str(len(raw_obj.get_data())))
        print("Number of samples: ", str(len(raw_obj.get_data()[0])))
        print("event_ids:", event_ids)
        print("events.shape:", events.shape)
        print("events:", events)

        # Print Epoch info
        data = epochs._data
        n_events = len(data) # or len(epochs.events)
        print("Number of events: " + str(n_events)) 
        n_channels = len(data[0,:]) # or len(epochs.ch_names)
        print("Number of channels: " + str(n_channels))
        n_times = len(data[0,0,:]) # or len(epochs.times)
        print("Number of time instances: " + str(n_times))
        print('Start time before the event' , epochs.tmin)
        print('Stop time after the event' , epochs.tmax)

        rest_epochs = epochs['T0']
        left_fist_epochs = epochs['T1']
        right_fist_epochs = epochs['T2']
        print("Number of T0 (Rest) events (epochs):", len(rest_epochs._data))
        print("Number of T1 events (epochs):", len(left_fist_epochs._data))
        print("Number of T2 events (epochs):", len(right_fist_epochs._data))


    if visualise:
        # Example plot of raw data
        plt.plot(raw_obj.get_data()[0][:4999])
        plt.title("Raw EEG, electrode 0, samples 0-4999")
        plt.show()

        # Plot electrode positions
        fig = raw_obj.plot_sensors(show_names=True)
        plt.show()

        # Power plot and MNE raw plot
        raw_obj.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
        raw_obj.plot(duration=5, n_channels=30);
        plt.show()

        # Plot epochs
        epochs.plot(scalings='auto')

        plt.plot(data[:,0,:].T)
        plt.title("Exemplar single-trial epoched data, for electrode 0")
        plt.show()

        epochs['T0'].average().plot()
        epochs['T1'].average().plot()
        epochs['T2'].average().plot()

        # Plot topomaps
        times = np.arange(0, 1, 0.1)
        epochs.average().plot_topomap(times, ch_type='eeg')
        epochs["T0"].average().plot_topomap(times, ch_type='eeg')
        epochs["T1"].average().plot_topomap(times, ch_type='eeg')
        epochs["T2"].average().plot_topomap(times, ch_type='eeg')

        plt.show()

    # -------------------------------------

if __name__ == "__main__":
    print("Running `data_processing.py` directly")
    # process_physionet(3, "S3-MI-FF", 4, verbose=True, visualise=True, save=False)

    task = 1
    for i in range(1, 25):
        process_physionet(i, "/task" + str(task) + "/s" + str(i), task)

    # process_physionet(3, "S3-MI-FF-8channel", 4)

