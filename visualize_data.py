import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

def save_data_to_hdf5(eeg_train, fmri_train, eeg_test, fmri_test, directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the full file path
    file_path = os.path.join(directory, 'eeg_fmri_data.h5')
    
    # Save data to HDF5 file
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('eeg_train', data=eeg_train)
        hf.create_dataset('fmri_train', data=fmri_train)
        hf.create_dataset('eeg_test', data=eeg_test)
        hf.create_dataset('fmri_test', data=fmri_test)
    
    print(f"Data saved successfully to {file_path}")

def plot_eeg_sample(eeg_data, sample_idx=0):
    sample = eeg_data[sample_idx]
    plt.figure(figsize=(15, 5))
    plt.imshow(sample[:, :, 0, 0].T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('EEG Sample {}'.format(sample_idx))
    plt.xlabel('Channels')
    plt.ylabel('Time/Frequency Bins')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_fmri_sample(fmri_data, sample_idx=0, time_idx=0):
    """
    Visualize the fMRI sample in a grid layout showing multiple slices.

    Parameters:
    fmri_data (np.array): The fMRI data.
    sample_idx (int): Index of the sample to visualize.
    time_idx (int): Time index to visualize.
    """
    sample = fmri_data[sample_idx, :, :, :, time_idx]
    
    fig, axes = plt.subplots(6, 5, figsize=(15, 15))
    fig.suptitle("fMRI Slices for Individual 1 at Time 0")
    
    for i in range(30):
        ax = axes[i // 5, i % 5]
        ax.imshow(sample[:, :, i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Slice {i}, Time {time_idx}')
    
    plt.tight_layout()
    plt.show()


def visualize_data(eeg_train, fmri_train, eeg_test, fmri_test):  
    # plot_eeg_sample(eeg_train, sample_idx=0)
    plot_fmri_sample(fmri_train, sample_idx=0)


