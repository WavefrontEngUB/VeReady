import time
import numpy as np
import h5py
from pyVeReady.utils.image_utils import *
from pyVeReady.microscopy_reconstruction.image_scanning_microscopy import *
from skimage.registration import phase_cross_correlation
from skimage.filters import gaussian
from scipy.ndimage import shift
import matplotlib.pyplot as plt

#%% LOAD DATA AND ARRANGE TO FLOAT64 STACK
start_total = time.time()

# Load ISM data and Extract raw dataset
start = time.time()
file = "Cell_ISM_dataset.h5"
data = h5py.File(file)

dset = data["data"]  # Shape: (rep, z, y, x, t, ch)
end = time.time()
print(f"File loaded in {end - start:.2f} s")

# Convert to float64
start = time.time()
dset_float = dset.astype(np.float64)
end = time.time()
print(f"Data type converted to float64 in {end - start:.2f} s")

# Integrate across reps, z, time
start = time.time()
image_stack = np.sum(dset_float, axis=(0, 1, 4))
end = time.time()
print(f"Image stack integrated in {end - start:.2f} s")

#%% ISM
# In here we have the starting point for ISM processing
# Having a 3D stack with the slicing (y, x, detector)

# Interactive viewer
viewer = Imshow3D(image_stack)
viewer.im.set_cmap('hot')
viewer.fig.canvas.draw_idle()

# Central detector estimation and Fingerprint visualization
start = time.time()
center_detector_index = estimate_central_detector(image_stack, True)
end = time.time()
print(f"Fingerprint visualized in {end - start:.2f} s")

# Compute shifts
start = time.time()
shifts, errors = compute_detector_shifts(image_stack, center_detector_index)
end = time.time()
print(f"Shift vectors computed in {end - start:.2f} s")

# Plot shift vectors
plot_shift_vectors(shifts)

# Apply reassignment shifts
start = time.time()
shifted_stack_full = apply_pixel_reassignment_shift(image_stack, shifts, shift_factor=1.0)
shifted_stack_half = apply_pixel_reassignment_shift(image_stack, shifts, shift_factor=0.5)
end = time.time()
print(f"Shifts applied in {end - start:.2f} s")

# Combine channels
start = time.time()
ism_image_full = np.sum(shifted_stack_full, axis=-1)
ism_image_half = np.sum(shifted_stack_half, axis=-1)
sum_all_detectors = np.sum(image_stack, axis=-1)
central_detector_image = image_stack[:, :, center_detector_index]
end = time.time()
print(f"Final images summed in {end - start:.2f} s")

# Plot all results
start = time.time()
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

axs[0].imshow(sum_all_detectors, cmap='hot')
axs[0].set_title('Sum All Detectors (Unshifted)')

axs[1].imshow(central_detector_image, cmap='hot')
axs[1].set_title('Central Detector (Closed Confocal)')

axs[2].imshow(ism_image_half, cmap='hot')
axs[2].set_title('Pixel Reassigned (Shift × 0.5)')

axs[3].imshow(ism_image_full, cmap='hot')
axs[3].set_title('Pixel Reassigned (Shift × 1)')

plt.show()
plt.tight_layout()
end = time.time()
print(f"Final images plotted in {end - start:.2f} s")

end_total = time.time()
print(f"Total runtime: {end_total - start_total:.2f} s")
