import numpy as np
import matplotlib.pyplot as plt

'''def save_data(train_data, test_data):
    print("main.py: On to saving and vizualization...")
    np.save('eeg_train.npy', train_data[0])
    np.save('fmri_train.npy', train_data[1])
    np.save('eeg_test.npy', test_data[0])
    np.save('fmri_test.npy', test_data[1])
    print("Data saved successfully.")'''

'''def load_data():
    eeg_train = np.load('eeg_train.npy')
    fmri_train = np.load('fmri_train.npy')
    eeg_test = np.load('eeg_test.npy')
    fmri_test = np.load('fmri_test.npy')
    return eeg_train, fmri_train, eeg_test, fmri_test'''

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
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("fMRI Slices for Individual 1 at Time 0")
    
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(sample[:, :, i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Slice {i}, Time {time_idx}')
    
    plt.tight_layout()
    plt.show()

def visualize_data(eeg_train, fmri_train, eeg_test, fmri_test):  
    plot_eeg_sample(eeg_train, sample_idx=0)
    plot_fmri_sample(fmri_train, sample_idx=0)


