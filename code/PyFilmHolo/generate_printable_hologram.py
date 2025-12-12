from PyFilmHolo.physical_constants import MM, UM, PDF_SECTOR_LENGTH, PRINTED_HOLOGRAM_SIZE, OPTIMAL_PIXEL_SIZE
from PyFilmHolo.hologram_computation import *
import skimage.data as data
from skimage.transform import rescale, resize
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

# Processing Parameters
binarize, threshold = True, 0.25
adjust_contrast, gamma = False, 2

# Load Target Image
target_im = plt.imread('UB_letras.png')

# If RGBA image convert to grayscale
if target_im.shape[2] > 1:
    target_im = 0.2126 * target_im[:,:,0] + 0.7152 * target_im[:,:,1] + 0.0722 * target_im[:,:,2]

# If necessary crop so it becomes a square image
# (Current implementation of GS algorithm is limited for square images, it could be upgraded though)
if target_im.shape[0] != target_im.shape[1]:
    # Implement here
    pass

if adjust_contrast:
    target_im = adjust_gamma(target_im, gamma)

# Normalize Image
target_im = target_im / target_im.max()

# Compute number of pixels in the hologram and resize target image accordingly
n_pixels = PRINTED_HOLOGRAM_SIZE // OPTIMAL_PIXEL_SIZE
target_im = resize(target_im, (n_pixels, n_pixels), order = 3)  # Order 3 means Bi-cubic Interpolation

# Binarize if desired
if binarize:
    target_im = np.array(target_im > threshold, dtype = np.float64)

print(f'Hologram Pixel Size in um: {OPTIMAL_PIXEL_SIZE / UM:.1f}')
print(f'Hologram Shape in mm: ({PRINTED_HOLOGRAM_SIZE / MM:.1f}, {PRINTED_HOLOGRAM_SIZE / MM:.1f})')
print(f'Hologram Shape in pixels: {target_im.shape}')

# Compute GS Hologram
c_holo_eff = compute_gerchberg_saxton_hologram(target_im, 'Quadratic', 125,
                                           2, False, 0.0)
plt.figure()
plt.imshow(target_im, cmap = 'gray')

# Compute Amplitude Pattern
holo_to_transmittance_method = 'Lee Phase'
match holo_to_transmittance_method:
    case 'Raw':
        transmittance_pattern = np.angle(c_holo_eff)
        transmittance_pattern[transmittance_pattern < 0] += 2 * np.pi
        transmittance_pattern /= 2 * np.pi
    case 'Lee Phase':
        transmittance_pattern = c_phase_hologram_to_transmittance_lee(c_holo_eff, 0)
        transmittance_pattern /= transmittance_pattern.max()

plt.figure()
plt.imshow(transmittance_pattern, cmap='gray')
