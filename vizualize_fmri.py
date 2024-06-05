import numpy as np
import matplotlib.pyplot as plt
from visualize_data import plot_fmri_sample, load_data, visualize_data  # Import the updated function

# Load the data and visualize
eeg_train, fmri_train, eeg_test, fmri_test = load_data()
visualize_data(eeg_train, fmri_train, eeg_test, fmri_test)
