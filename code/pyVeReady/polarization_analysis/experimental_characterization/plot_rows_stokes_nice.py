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
        h, w = img.shape  # Get image dimensions

        # Define crop boundaries
        row_start = max(0, row_center - crop_size // 2)
        row_end = min(h, row_center + crop_size // 2)
        col_start = max(0, col_center - crop_size // 2)
        col_end = min(w, col_center + crop_size // 2)

        # Crop and store
        cropped_img = img[row_start:row_end, col_start:col_end]
        cropped_list.append(cropped_img)

    return cropped_list

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12
})

# Load Laguerre-Gauss Right
lg_right_path_tiff = ask_files_location('Laguerre-Gauss Right', return_first_string=True)
data_list_lg = load_tiff_images_to_list(lg_right_path_tiff, slicing_index = 0)
row_center, col_center = find_beam_center(np.max(np.array(data_list_lg), axis=0))
data_list_lg = crop_images(data_list_lg, int(row_center), int(col_center), crop_size=20)
s0_data_lg, s1_data_lg, s2_data_lg, s3_data_lg = compute_stokes(data_list_lg, 'Rotating QWP')
s1_data_lg = s1_data_lg / s0_data_lg
s2_data_lg = s2_data_lg / s0_data_lg
s3_data_lg = s3_data_lg / s0_data_lg
s0_data_lg = s0_data_lg / s0_data_lg.max()

# Load Azimuthal
az_path_tiff = ask_files_location('Azimuthal', return_first_string=True)
data_list_az = load_tiff_images_to_list(az_path_tiff, slicing_index = 0)
row_centeraz, col_centeraz = find_beam_center(np.max(np.array(data_list_az), axis=0))
data_list_az = crop_images(data_list_az, int(row_centeraz), int(col_centeraz), crop_size=20)
s0_data_az, s1_data_az, s2_data_az, s3_data_az = compute_stokes(data_list_az, 'Rotating QWP')
s1_data_az = s1_data_az / s0_data_az
s2_data_az = s2_data_az / s0_data_az
s3_data_az = s3_data_az / s0_data_az
s0_data_az = s0_data_az / s0_data_az.max()

X, Y, r, phi = pixel_coordinates(s0_data_az)
pixel_size_um = 6
extent_lims = [X.min()*pixel_size_um, X.max()*pixel_size_um, Y.min()*pixel_size_um, Y.max()*pixel_size_um]

# Plots
fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)

# Increase horizontal spacing between columns
fig.subplots_adjust(wspace=0.5)  # Wider spacing

# Define colormap limits
vmin, vmax = -1, 1  # For S1, S2, S3
cmap = 'bwr'  # Blue-White-Red for polarization

# Plot S0 (grayscale)
for i, (data, label) in enumerate([(s0_data_lg, r'LG Circular Right'), (s0_data_az, r'Azimuthal')]):
    ax = axes[i, 0]
    im0 = ax.imshow(data, cmap='gray', extent=extent_lims)
    ax.set_title(r'$S_0$')
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.text(-55, -55, label, color='white', fontsize=12)

# Add separate colorbar for S0
cbar0 = fig.colorbar(im0, ax=axes[:, 0], fraction=0.1, pad=0.08, aspect=30)  # Increased pad for separation

# Plot S1, S2, S3 with red-blue colormap
for i, (S1, S2, S3) in enumerate([(s1_data_lg, s2_data_lg, s3_data_lg), (s1_data_az, s2_data_az, s3_data_az)]):
    for j, (data, title) in enumerate([(S1, r'$S_1$'), (S2, r'$S_2$'), (S3, r'$S_3$')]):
        ax = axes[i, j+1]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_lims)
        ax.set_title(title)
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')

# Add shared colorbar for S1-S3
cbar = fig.colorbar(im, ax=axes[:, 1:], fraction=0.1, pad=0.02, aspect=30)  # No extra pad here

plt.show()
