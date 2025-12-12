import numpy as np
import matplotlib.pyplot as plt
from pyVeReady.polarization_analysis.polarization_analysis_functions import compute_stokes, find_beam_center
from pyVeReady.utils.paths_utils import ask_files_location
from pyVeReady.utils.image_utils import *
import os
import scienceplots
plt.style.use(['science','no-latex'])


def process_file(filepath):
    """ Process a single file to compute S0, find beam center, and extract line profiles """
    data_list = load_tiff_images_to_list([filepath], slicing_index=0)
    s0_data, s1_data, s2_data, s3_data = compute_stokes(data_list, 'Rotating QWP')
    row, col = find_beam_center(s0_data)
    row, col = int(round(row)), int(round(col))  # Ensure row and col are properly rounded integers
    roi_size = 22

    # Ensure cropping does not exceed image bounds
    half_roi = roi_size // 2
    row_start, row_end = max(0, row - half_roi), min(s0_data.shape[0], row + half_roi)
    col_start, col_end = max(0, col - half_roi), min(s0_data.shape[1], col + half_roi)
    s0_data = s0_data[row_start:row_end, col_start:col_end]

    # Adjust center relative to cropped region
    s0_data = s0_data / np.max(s0_data)
    s0_data = s0_data * 255
    row, col = find_beam_hole_center(s0_data, roi_size=22, show_estimation=False)
    row, col = int(round(row)), int(round(col))  # Ensure row and col are properly rounded integers
    roi_size = 22

    # Ensure cropping does not exceed image bounds
    half_roi = roi_size // 2
    row_start, row_end = max(0, row - half_roi), min(s0_data.shape[0], row + half_roi)
    col_start, col_end = max(0, col - half_roi), min(s0_data.shape[1], col + half_roi)
    s0_data = s0_data[row_start:row_end, col_start:col_end]

    row_center = row - row_start
    col_center = col - col_start

    x_profile = s0_data[row_center, :]
    y_profile = s0_data[:, col_center]

    x_profile = x_profile / np.max(x_profile)
    y_profile = y_profile / np.max(y_profile)

    return filepath, s0_data, (row_center, col_center), x_profile, y_profile


def plot_profiles(profiles, title, xlabel, ylabel):
    """ Plot multiple line profiles on the same figure """
    plt.figure(figsize=(10, 6))

    # Saturated colors (Red, Green, Blue, Cyan, Magenta, Yellow, Black)
    saturated_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # Loop through profiles and plot them using the saturated colors
    for i, (filename, profile) in enumerate(profiles):
        plt.plot(profile, label=os.path.basename(filename), color=saturated_colors[i % len(saturated_colors)], lw=2)

    # Customize the plot
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(title='Files', loc='upper right', fontsize=12)

    # Make sure the ticks are readable and well-spaced
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()


def show_s0_image(s0_data, beam_center, filename):
    """ Show the S0 image with a marker indicating the beam center """
    plt.figure(figsize=(8, 8))
    plt.imshow(s0_data, cmap='inferno')
    plt.scatter(beam_center[1], beam_center[0], color='cyan', marker='+', s=100, label='Beam Center')
    plt.colorbar(label='S0 Intensity')
    plt.title(f'S0 Image - {os.path.basename(filename)}', fontsize=16)
    plt.legend(fontsize=12)
    plt.axis('off')  # Turn off axis for a cleaner look
    plt.tight_layout()
    plt.show()


# Select and process files
tiff_filepaths = ask_files_location('Select Tiff Files')

x_profiles = []
y_profiles = []

for filepath in tiff_filepaths:
    filename, s0_image, beam_center, x_profile, y_profile = process_file(filepath)
    x_profiles.append((filename, x_profile))
    y_profiles.append((filename, y_profile))
    show_s0_image(s0_image, beam_center, filename)

# Plot all line profiles
plot_profiles(x_profiles, 'X-Line Profiles', 'X Position', 'S0 Intensity')
plot_profiles(y_profiles, 'Y-Line Profiles', 'Y Position', 'S0 Intensity')
