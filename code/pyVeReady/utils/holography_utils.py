import numpy as np
from skimage.transform import rescale, resize
from pyVeReady.utils.image_utils import *

# Functions for computing propagation of the electric field
def centered_fft2(array, energy_preservation=False):
    """
    Computes the centered 2D FFT of the given array.

    Parameters:
        array (numpy.ndarray): 2D array representing the aperture function.
        energy_preservation(bool, optional): Preserves the energy of the output Fourier transform.

    Returns:
        numpy.ndarray: Centered Fourier transform of the array.
    """
    ft_array = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array)))
    if energy_preservation:
        ft_array /= np.sqrt(ft_array.size)

    return ft_array


def centered_ifft2(array, energy_preservation=False):
    """
    Computes the centered 2D IFFT of the given array.

    Parameters:
        array (numpy.ndarray): 2D array representing the aperture function.
        energy_preservation(bool, optional): Preserves the energy of the output Fourier transform.

    Returns:
        numpy.ndarray: Centered Inverse Fourier transform of the array.
    """
    ift_array = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(array)))
    if energy_preservation:
        ift_array *= np.sqrt(ift_array.size)

    return ift_array


def compute_image_plane_resolution(wavelength, slm_pixel_pitch, hologram_pixels):
    """
    Computes the image plane resolution from a pixelated hologram displayed on an SLM.
    Assumes a square hologram is used.

    Parameters:
        wavelength (float): Wavelength of light.
        slm_pixel_pitch (float): Pixel pitch of the SLM in meters.
        hologram_pixels (int): Number of pixels in the hologram.

    Returns:
        float: Resolution at the image plane in reciprocal meters.
    """
    maximum_deflected_position = wavelength / (2 * slm_pixel_pitch)
    resolution_at_image_plane = maximum_deflected_position / hologram_pixels
    return resolution_at_image_plane


def compute_total_energy(c_field):
    """
    Computes the total energy of a complex field. The total energy is calculated as the sum of the squared amplitudes
    of all elements in the input array.

    Parameters:
        c_field (numpy.ndarray): A 2D complex array representing an electric field.

    Returns:
        float: The total energy of the field.
    """
    energy = np.sum(np.abs(c_field)**2)
    return energy


# Functions that create Aperture Masks
def centered_triangle(rows, cols, base, height):
    """
    Creates a 2D array with a centered triangular aperture of ones, the rest being zeros.
    """
    array = np.zeros((rows, cols), dtype=int)
    center_x, center_y = cols // 2, rows // 2
    for y in range(height):
        width = int(base * (1 - y / height))
        start_x = center_x - width // 2
        end_x = center_x + width // 2
        array[center_y - height // 2 + y, start_x:end_x] = 1
    return array


def centered_circle(rows, cols, radius):
    """
    Creates a 2D array with a centered circular aperture of ones, the rest being zeros.

    Parameters:
        rows (int): Number of rows in the output array.
        cols (int): Number of columns in the output array.
        radius (int): Radius of the circular aperture.

    Returns:
        np.ndarray: 2D array with a centered circular aperture of ones.
    """
    array = np.zeros((rows, cols), dtype=int)

    y, x = np.ogrid[:rows, :cols]
    center_y, center_x = rows // 2, cols // 2
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    array[mask] = 1

    return array

def centered_rectangle(rows, cols, rect_height, rect_width):
    """
    Creates a 2D array with a centered rectangle of ones, the rest being zeros.

    Parameters:
        rows (int): Number of rows in the output array.
        cols (int): Number of columns in the output array.
        rect_height (int): Height of the rectangle of ones.
        rect_width (int): Width of the rectangle of ones.

    Returns:
        np.ndarray: 2D array with a centered rectangle of ones.
    """
    array = np.zeros((rows, cols), dtype=int)

    start_row = (rows - rect_height) // 2
    start_col = (cols - rect_width) // 2

    array[start_row:start_row + rect_height, start_col:start_col + rect_width] = 1

    return array

# Misc

def create_noise_area(target_im, noise_area_factor):
    """
    Adds a noise area to an image and creates a corresponding mask.
    The output image containing the noise area has the same shape as
    the input image target_im.

    The function downscales the input image based on the `noise_area_factor`,
    pads it back to the original size, and normalizes the values. It also
    generates a mask where the signal area is marked with ones and the noise
    area with zeros.

    Parameters:
        target_im (numpy.ndarray): The input image.
        noise_area_factor (float): The factor determining the size of the noise area.

    Returns:
        tuple: The image with added noise area and the signal area mask.
    """
    original_im, original_shape = target_im[:,:], target_im.shape
    downscaled_im = rescale(original_im, 1 - noise_area_factor)  # Downsample image to allow for noise area
    im_with_noise_area = pad2size(downscaled_im, original_shape[0], original_shape[1])  # Fill with noise area to obtain original shape
    im_with_noise_area = im_with_noise_area / im_with_noise_area.max()  # Normalize image
    mask_signal_area = pad2size(np.ones_like(downscaled_im, dtype=bool), *original_shape)  # Create mask, 1 for signal area and 0 for noise area
    return im_with_noise_area, mask_signal_area, downscaled_im

def c_phase_hologram_to_transmittance_lee(c_hologram, lin_phase_freq):
    X, Y, _, _ = pixel_coordinates(c_hologram)
    transmittance_function = 1/2 * (1 + np.cos(2*np.pi * (X-Y) * lin_phase_freq - np.angle(c_hologram)))
    return transmittance_function

def compute_initial_phase(array_shape, phase_type='Random'):
    match phase_type:
        case 'Random':
            phase0 = 2 * np.pi * np.random.rand(*array_shape)
        case 'Quadratic':
            x = np.arange(array_shape[1]) - array_shape[1] // 2
            y = np.arange(array_shape[0]) - array_shape[0] // 2
            X, Y = np.meshgrid(x, y)
            k = np.pi / (1 * array_shape[0]) * 4
            phase0 = k * (X ** 2) + k * (Y ** 2)
        case _:
            print('Unknown initial phase, random initial phase generated')
            phase0 = 2 * np.pi * np.random.rand(*array_shape)
    return phase0

def generate_points(size, spacing, span, rows_offset=0, cols_offset=0):
    """
    Generates a square 2D array with ones evenly spaced in both directions within a square span window,
    allowing for row and column offsets.

    Parameters:
        size (int): The size (size x size) of the square array.
        spacing (int): The spacing between ones.
        span (int): The size of the square region where ones are placed.
        rows_offset (int): Offset in rows for displacement.
        cols_offset (int): Offset in columns for displacement.

    Returns:
        np.ndarray: A (size x size) array with ones placed in a grid pattern within the span window.
    """
    array = np.zeros((size, size), dtype=int)
    center = size // 2
    half_span = span // 2

    x_start, x_end = center - half_span + rows_offset, center + half_span + rows_offset
    y_start, y_end = center - half_span + cols_offset, center + half_span + cols_offset

    for i in range(x_start, x_end + 1, spacing):
        for j in range(y_start, y_end + 1, spacing):
            if 0 <= i < size and 0 <= j < size:
                array[i, j] = 1

    return array