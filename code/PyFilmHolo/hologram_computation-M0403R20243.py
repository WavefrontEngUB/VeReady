import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import rescale, resize
import matplotlib
from pyVeReady.utils.image_utils import *
matplotlib.use('tkagg')
np.random.seed(27)

def good_fft2(array, energy_preservation = False):
    ft_array = fftshift(fft2(ifftshift(array)))
    if energy_preservation:
        return ft_array / np.sqrt(ft_array.size)
    else:
        return ft_array
def good_ifft2(array, energy_preservation = False):
    ift_array = fftshift(ifft2(ifftshift(array)))
    if energy_preservation:
        return ift_array * np.sqrt(ift_array.size)
    else:
        return ift_array

def create_square_mask(image_size, holo_size):
    """
    Creates a square binary mask with a centered square of ones.

    Parameters:
    image_size (tuple): Size of the full image (height, width).
    holo_size (tuple): Size of the square region of ones (height, width).

    Returns:
    ndarray: Binary mask with a centered square of ones.
    """
    mask = np.zeros(image_size, dtype = np.uint8)
    start_x, start_y = (np.array(image_size) - np.array(holo_size)) // 2
    mask[start_x:start_x + holo_size[0], start_y:start_y + holo_size[1]] = 1
    return mask

def compute_total_energy(c_field):
    """"
    To be documented
    """
    energy = np.sum(np.abs(c_field)**2)
    return energy

def update_plot(holo_field, focal_field, iteration, fig, axes):
    """
    Updates the real-time plots of the hologram plane and focal field phase.
    """
    fig.suptitle(f'Iteration {iteration}')
    axes[0].cla()
    axes[0].imshow(np.angle(holo_field), cmap = 'twilight')
    axes[0].set_title("Hologram Plane Phase")
    axes[1].cla()
    axes[1].imshow(np.abs(focal_field)**2, cmap = 'gray')
    axes[1].set_title("Focal Field Intensity")
    plt.pause(0.001)

def create_noise_area(target_im, noise_area_factor):
    original_im, original_shape = target_im[:,:], target_im.shape
    downscaled_im = rescale(original_im, 1 - noise_area_factor)  # Downsample image to allow for noise area
    im_with_noise_area = pad2size(downscaled_im, original_shape[0], original_shape[1])  # Fill with noise area to obtain original shape
    im_with_noise_area = im_with_noise_area / im_with_noise_area.max()  # Normalize image
    mask_signal_area = pad2size(np.ones_like(downscaled_im), original_shape[0], original_shape[1])  # Create mask, 1 for signal area and 0 for noise area
    return im_with_noise_area, mask_signal_area

def compute_gerchberg_saxton_hologram(target_im, initial_phase, n_iterations,
                                      holo_padded_size_factor, use_noise_area, noise_area_factor):
    holo_shape = target_im.shape

    # Modify target image to allow for a noise area if desired
    target_im, mask_signal_area = create_noise_area(target_im, noise_area_factor)

    # Upsample target image if hologram is zero padded (to obtain higher resolution in the focal plane)
    target_im = rescale(target_im, holo_padded_size_factor)
    mask_holo = create_square_mask(target_im.shape, holo_shape)

    # Compute Amplitude and Initial Phase of the electric field at the Focal Plane
    target_ampl = np.sqrt(target_im)
    match initial_phase:
        case 'Random':
            phase0 = 2 * np.pi * np.random.rand(*target_ampl.shape)
        case 'Quadratic':
            x = np.arange(target_ampl.shape[1]) - target_ampl.shape[1] // 2
            y = np.arange(target_ampl.shape[0]) - target_ampl.shape[0] // 2
            X, Y = np.meshgrid(x, y)
            k = np.pi / (1 * 2 * target_ampl.shape[0])
            phase0 = k * X**2 + k * Y**2
        case _:
            print('Unknown initial phase, random initial phase generated')
            phase0 = 2 * np.pi * np.random.rand(*target_ampl.shape)

    plt.ion()  # Turn on interactive mode for real-time updates
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    c_focal_field = target_ampl * np.exp(1j * phase0)  # Initialize Field at Focal Plane
    total_energy = compute_total_energy(c_focal_field)
    for iteration in range(n_iterations):
        c_hologram_field = good_ifft2(c_focal_field)  # Propagate to Hologram Plane
        c_hologram_field = np.exp(1j * np.angle(c_hologram_field))  # Apply Phase-only Constraint
        c_hologram_field[mask_holo == 0] = 0  # Preserve hologram zero padding (Bandwidth Constraint)

        c_focal_field = good_fft2(c_hologram_field)  # Propagate to Focal Plane
        update_plot(c_hologram_field, c_focal_field, iteration + 1, fig, axes)

        if iteration != n_iterations-1:
            if use_noise_area:
                c_focal_field = target_ampl * np.exp(1j * np.angle(c_focal_field))  # Apply Target Amplitude Constraint

            else:
                c_focal_field = target_ampl * np.exp(1j * np.angle(c_focal_field))  # Apply Target Amplitude Constraint

    c_holo_eff = c_hologram_field[mask_holo == 1].reshape(holo_shape)
    return c_holo_eff

def c_phase_hologram_to_transmittance_lee(c_hologram, lin_phase_freq):
    X, Y, _, _ = pixel_coordinates(c_hologram)
    transmittance_function = 1/2 * (1 + np.cos(2*np.pi * (X-Y) * lin_phase_freq - np.angle(c_hologram)))
    return transmittance_function

if __name__ == '__main__':
    # Load Target Image
    target_im = data.camera().astype(np.float64)

    # Gerchberg-Saxton Parameters
    holo_padded_size_factor = 2
    noise_area_factor = 0
    initial_phase = 'Quadratic'
    use_noise_area = True
    n_iterations = 50

    #c_holo_eff = compute_gerchberg_saxton_hologram(target_im, initial_phase, n_iterations,
     #                                          holo_padded_size_factor, use_noise_area, noise_area_factor)

