import time
import numpy as np
import h5py
from pyVeReady.utils.image_utils import *
from skimage.registration import phase_cross_correlation
from skimage.filters import gaussian
from scipy.ndimage import shift
import matplotlib.pyplot as plt


def apply_pixel_reassignment_shift(image_stack, shift_vectors, shift_factor=1.0):
    """
    Shift each channel of the input image stack based on the provided shift vectors using interpolation.

    This function applies subpixel shifts to each channel independently. Shifts are scaled by
    `shift_factor` and applied using interpolation (via `scipy.ndimage.shift`), which avoids artifacts
    from integer-only translations.

    Parameters:
        image_stack (ndarray): 3D array of shape (Y, X, channels), representing the image stack.
        shift_vectors (ndarray): Array of shape (channels, 2), containing (dy, dx) shift values.
        shift_factor (float): Multiplier applied to each shift vector before shifting.

    Returns:
        ndarray: Shifted image stack with the same shape as the input.
    """
    shifted_stack = np.empty_like(image_stack)
    scaled_shifts = shift_vectors * shift_factor
    for ch_idx in range(image_stack.shape[2]):
        shifted_stack[:, :, ch_idx] = shift(image_stack[:, :, ch_idx], scaled_shifts[ch_idx, :])
    return shifted_stack


def compute_detector_shifts(image_stack, center_detector_index, upsample_factor=100, apodize=True, filter_sigma=1):
    """
    Compute relative shifts between detector images using phase cross-correlation.

    Parameters:
        image_stack (ndarray): 3D array of shape (Y, X, channels), the image data.
        center_detector_index (int): Index of the reference detector channel.
        upsample_factor (int): Upsampling factor for subpixel registration accuracy.
        apodize (bool): Whether to apply a 2D Hann window to reduce edge effects.
        filter_sigma (float): Standard deviation for optional Gaussian smoothing.

    Returns:
        shifts (ndarray): Array of shape (channels, 2) with (dy, dx) shifts for each detector.
        errors (ndarray): Registration error for each detector relative to the reference.
    """
    Ny, Nx, n_channels = image_stack.shape

    if apodize:
        window = hann2d((Ny, Nx))[:, :, None]
        image_stack = image_stack * window

    if filter_sigma > 0:
        image_stack = gaussian(image_stack, sigma=filter_sigma, channel_axis=-1)

    reference_image = image_stack[:, :, center_detector_index]
    shifts = np.zeros((n_channels, 2))
    errors = np.zeros(n_channels)

    for ch_idx in range(n_channels):
        shift_, error, _ = phase_cross_correlation(
            reference_image, image_stack[:, :, ch_idx], upsample_factor=upsample_factor, normalization=None
        )
        shifts[ch_idx] = shift_
        errors[ch_idx] = error

    return shifts, errors


def plot_shift_vectors(shifts):
    """
    Plot shift vectors as a 2D scatter plot with index annotations.

    Parameters:
        shifts (ndarray): Array of shape (channels, 2) containing (dy, dx) vectors.
    """
    dy = shifts[:, 0]
    dx = shifts[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(dx, dy, color='blue')

    for i in range(len(shifts)):
        plt.text(dx[i], dy[i], str(i), fontsize=8, ha='right', va='bottom')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('dx (pixels)')
    plt.ylabel('dy (pixels)')
    plt.title('Shift Vectors Scatter Plot')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()


def hann2d(shape):
    """
    Generate a 2D Hann window for apodization.

    Parameters:
        shape (tuple): The shape of the output window (Ny, Nx, ...).

    Returns:
        ndarray: 2D Hann window of the given shape.
    """
    Ny, Nx = shape[0], shape[1]
    y = np.arange(Ny)
    x = np.arange(Nx)
    Y, X = np.meshgrid(y, x, indexing='ij')
    Wy = 0.5 * (1 - np.cos((2 * np.pi * Y) / (Ny - 1)))
    Wx = 0.5 * (1 - np.cos((2 * np.pi * X) / (Nx - 1)))
    return Wy * Wx

def estimate_central_detector(image_stack, show_fingerprint=True):
    """
    Estimate the central detector as the one with highest total signal.

    Optionally displays the detector fingerprint with an 'X' marking the estimated center.

    Parameters:
        image_stack (ndarray): 3D image stack (Y, X, detectors).
        show_fingerprint (bool): If True, shows fingerprint plot.

    Returns:
        int: Index of the estimated central detector.
    """
    flat_fingerprint = np.sum(image_stack, axis=(0, 1))
    side = int(np.sqrt(flat_fingerprint.shape[0]))
    fingerprint = flat_fingerprint.reshape((side, side))
    max_idx = np.argmax(flat_fingerprint)
    center_y, center_x = divmod(max_idx, side)

    if show_fingerprint:
        plt.figure()
        plt.imshow(fingerprint, cmap='hot')
        plt.plot(center_x, center_y, 'bx', markersize=12, mew=2)
        plt.title("Estimated Central Detector")
        plt.colorbar(label='Integrated Signal')
        plt.show()

    return max_idx