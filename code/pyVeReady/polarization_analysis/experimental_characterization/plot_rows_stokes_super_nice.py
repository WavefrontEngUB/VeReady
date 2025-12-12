import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from pyVeReady.utils.paths_utils import *
from pyVeReady.utils.image_utils import *
from pyVeReady.polarization_analysis.polarization_analysis_functions import *
from pyVeReady.measure_scripts.beam_focal_scan_linear_phase.analyze_datasets import *
from scipy.ndimage import center_of_mass
matplotlib.use('TkAgg')

def plot_polarization(rows_data, extents, labels):
    """
    Plots polarization data (S0, S1, S2, S3) for multiple rows.

    Parameters:
        rows_data (list of lists): Each inner list should contain [S0, S1, S2, S3] data arrays.
        extents (list): List of extent limits for each row.
        labels (list): List of labels for each row.
    """
    n_rows = len(rows_data)
    n_cols = 4  # S0, S1, S2, S3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3 * n_rows), constrained_layout=True)
    fig.subplots_adjust(wspace=0.5)  # Increase horizontal spacing

    # Define colormap limits
    vmin, vmax = -1, 1  # For S1, S2, S3
    cmap = 'bwr'  # Blue-White-Red for polarization

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure consistent indexing

    # Plot each row
    for i, (row, extent, label) in enumerate(zip(rows_data, extents, labels)):

        # Plot S0 (grayscale)
        ax0 = axes[i, 0]
        im0 = ax0.imshow(row[0], cmap='gray', extent=extent)
        ax0.set_title(r'$S_0$')
        ax0.set_xlabel('x (μm)')
        ax0.set_ylabel('y (μm)')
        ax0.text(-55, -55, label, color='white', fontsize=12)

        # Plot S1, S2, S3 with red-blue colormap
        for j, (data, title) in enumerate([(row[1], r'$S_1/S_0$'), (row[2], r'$S_2/S_0$'), (row[3], r'$S_3/S0$')]):
            ax = axes[i, j + 1]
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
            ax.set_title(title)
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')

    # Add colorbars
    fig.colorbar(im0, ax=axes[:, 0], fraction=0.1, pad=0.08, aspect=30)  # S0 colorbar
    fig.colorbar(im, ax=axes[:, 1:], fraction=0.1, pad=0.02, aspect=30)  # S1-S3 colorbar

    plt.show()

if __name__ == '__main__':
    # Example usage
    labels = ['LG Circular Right', 'Azimuthal', 'Azimuthal Scan']
    data_beam_one = load_tiff_images_to_list(ask_files_location(labels[0], return_first_string=True))
    data_beam_two = load_tiff_images_to_list(ask_files_location(labels[1], return_first_string=True))

    dictionary_scan_filepath = ask_files_location(labels[2], return_first_string=True)
    with open(dictionary_scan_filepath, 'rb') as file:
        dictionary = pickle.load(file)
        # Unpack values from dictionary
        list_of_all_measures = dictionary['List of All Measures']

    s0_scan, s1_scan, s2_scan, s3_scan = compute_maximum_projection_stokes_from_scan(list_of_all_measures)

    row_c_one, col_c_one = find_beam_center(np.max(np.array(data_beam_one), axis=0))
    row_c_two, col_c_two = find_beam_center(np.max(np.array(data_beam_two), axis=0))
    row_c_scan , col_c_scan = center_of_mass(s0_scan>80)

    roi_size = 20
    data_beam_one = [crop_image(im_beam, row_c_one, col_c_one, roi_size) for im_beam in data_beam_one]
    data_beam_two = [crop_image(im_beam, row_c_two, col_c_two, roi_size) for im_beam in data_beam_two]

    roi_scan = 145
    stokes_scan = [crop_image(im, row_c_scan, col_c_scan, roi_scan) for im in (s0_scan, s1_scan, s2_scan, s3_scan)]
    # stokes_scan = [s0_scan, s1_scan, s2_scan, s3_scan]

    camera_pixel_size_um = 6
    X_one, Y_one, _, _ = pixel_coordinates(data_beam_one[0])
    X_one, Y_one = X_one * camera_pixel_size_um, Y_one * camera_pixel_size_um
    X_scan, Y_scan, _, _ = pixel_coordinates(stokes_scan[0])
    X_scan, Y_scan = X_scan * camera_pixel_size_um, Y_scan * camera_pixel_size_um

    extents = [[X_one.min(), X_one.max(), Y_one.min(), Y_one.max()],
               [X_one.min(), X_one.max(), Y_one.min(), Y_one.max()],
               [X_scan.min(), X_scan.max(), Y_scan.min(), Y_scan.max()],]

    stokes_data = [
        [*compute_stokes(data_beam_one, 'Rotating QWP')],
        [*compute_stokes(data_beam_two, 'Rotating QWP')],
        stokes_scan
    ]

    normalized_data = [[s0/s0.max(), s1/s0, s2/s0, s3/s0] for [s0, s1, s2, s3] in stokes_data]

    plot_polarization(normalized_data, extents, labels)
