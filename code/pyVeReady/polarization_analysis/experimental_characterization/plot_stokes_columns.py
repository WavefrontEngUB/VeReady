import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')
from pyVeReady.utils.paths_utils import *
from pyVeReady.utils.image_utils import *
from pyVeReady.polarization_analysis.polarization_analysis_functions import *


def crop_images(image_list, row_center, col_center, crop_size=100):
    """Crops each image in the list around the given center."""
    cropped_list = []

    for img in image_list:
        h, w = img.shape
        row_start = max(0, row_center - crop_size // 2)
        row_end = min(h, row_center + crop_size // 2)
        col_start = max(0, col_center - crop_size // 2)
        col_end = min(w, col_center + crop_size // 2)
        cropped_img = img[row_start:row_end, col_start:col_end]
        cropped_list.append(cropped_img)

    return cropped_list


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12
})

# Load Laguerre-Gauss Right
lg_right_path_tiff = ask_files_location('Laguerre-Gauss Right', return_first_string=True)
data_list_lg = load_tiff_images_to_list(lg_right_path_tiff, slicing_index=0)
row_center, col_center = find_beam_center(np.max(np.array(data_list_lg), axis=0))
data_list_lg = crop_images(data_list_lg, int(row_center), int(col_center), crop_size=20)
s0_data_lg, s1_data_lg, s2_data_lg, s3_data_lg = compute_stokes(data_list_lg, 'Rotating QWP')
s1_data_lg /= s0_data_lg
s2_data_lg /= s0_data_lg
s3_data_lg /= s0_data_lg
s0_data_lg /= s0_data_lg.max()

# Load Azimuthal
az_path_tiff = ask_files_location('Azimuthal', return_first_string=True)
data_list_az = load_tiff_images_to_list(az_path_tiff, slicing_index=0)
row_centeraz, col_centeraz = find_beam_center(np.max(np.array(data_list_az), axis=0))
data_list_az = crop_images(data_list_az, int(row_centeraz), int(col_centeraz), crop_size=20)
s0_data_az, s1_data_az, s2_data_az, s3_data_az = compute_stokes(data_list_az, 'Rotating QWP')
s1_data_az /= s0_data_az
s2_data_az /= s0_data_az
s3_data_az /= s0_data_az
s0_data_az /= s0_data_az.max()

X, Y, r, phi = pixel_coordinates(s0_data_az)
pixel_size_um = 6
extent_lims = [X.min() * pixel_size_um, X.max() * pixel_size_um, Y.min() * pixel_size_um, Y.max() * pixel_size_um]

# Plots in columns
fig, axes = plt.subplots(4, 2, figsize=(9, 12), constrained_layout=False)
fig.subplots_adjust(wspace=0.03)  # Reduce spacing between columns

vmin, vmax = -1, 1
cmap = 'bwr'

# Define labels for rows
stokes_labels = [r'$S_0$', r'$S_1$', r'$S_2$', r'$S_3$']
beam_labels = [r'LG Circular Right', r'Azimuthal']
ims = []

# Plot Stokes parameters
for j, (data_lg, data_az) in enumerate([(s0_data_lg, s0_data_az),
                                        (s1_data_lg, s1_data_az),
                                        (s2_data_lg, s2_data_az),
                                        (s3_data_lg, s3_data_az)]):
    for i, data in enumerate([data_lg, data_az]):
        ax = axes[j, i]
        im = ax.imshow(data, cmap='gray' if j == 0 else cmap, vmin=None if j == 0 else vmin,
                       vmax=None if j == 0 else vmax, extent=extent_lims)
        ims.append(im)
        ax.set_title(stokes_labels[j])
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        if j == 0:
            ax.text(-55, -55, beam_labels[i], color='white', fontsize=12)

# Add separate colorbars on the right
cbar_ax1 = fig.add_axes([0.92, 0.78, 0.02, 0.14])  # Adjusted for S0
fig.colorbar(ims[0], cax=cbar_ax1)

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.58])  # Adjusted for S1-S3
fig.colorbar(ims[-1], cax=cbar_ax2)

plt.show()
