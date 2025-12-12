# import numpy as np
# from numpy.fft import fft2, ifft2, fftshift, ifftshift
# import matplotlib.pyplot as plt
# from skimage import data
# from skimage.transform import rescale, resize
# import matplotlib
# from pyVeReady.utils.image_utils import *
#
# # Use TkAgg backend for Matplotlib
# matplotlib.use('tkagg')
# np.random.seed(27)
#
# def c_phase_hologram_to_transmittance_lee(c_hologram, deflection_spatial_frequency):
#     """
#     Converts a phase hologram to a transmittance profile using Lee's method.
#
#     Parameters:
#         c_hologram (numpy.ndarray): Input complex hologram.
#         deflection_spatial_frequency (float): Deflection spatial frequency scaling the phase difference.
#
#     Returns:
#         numpy.ndarray: Transmittance profile.
#     """
#     X, Y, _, _ = pixel_coordinates(c_hologram, 0, 0)
#     return 0.5 * (1 + np.cos(2*np.pi * (X - Y) * deflection_spatial_frequency - np.angle(c_hologram)))
#
#
# def good_fft2(array, energy_preservation=False):
#     """
#     Computes the centered 2D Fourier Transform of an array.
#
#     Parameters:
#         array (numpy.ndarray): Input array.
#         energy_preservation (bool): If True, scales output for energy preservation.
#
#     Returns:
#         ft_array (numpy.ndarray): Fourier transformed array.
#     """
#     ft_array = fftshift(fft2(fftshift(array)))
#     return ft_array / np.sqrt(ft_array.size) if energy_preservation else ft_array
#
#
# def good_ifft2(array, energy_preservation=False):
#     """
#     Computes the centered 2D Inverse Fourier Transform of an array.
#
#     Parameters:
#         array (numpy.ndarray): Input array.
#         energy_preservation (bool): If True, scales output for energy preservation.
#
#     Returns:
#         ift_array (numpy.ndarray): Inverse Fourier transformed array.
#     """
#     ift_array = fftshift(ifft2(fftshift(array)))
#     return ift_array * np.sqrt(ift_array.size) if energy_preservation else ift_array
#
#
# def create_square_mask(image_size, holo_size):
#     """
#     Creates a square boolean mask with a centered square of True values.
#
#     Parameters:
#         image_size (tuple): Size of the full image (height, width).
#         holo_size (tuple): Size of the square region of True values (height, width).
#
#     Returns:
#         mask (numpy.ndarray): Boolean mask with a centered square of True values.
#     """
#     mask = np.zeros(image_size, dtype=bool)
#     start_x, start_y = (np.array(image_size) - np.array(holo_size)) // 2
#     mask[start_x:start_x + holo_size[0], start_y:start_y + holo_size[1]] = True
#     return mask
#
#
# def compute_total_energy(c_field):
#     """
#     Computes the total energy of a complex field.
#
#     Parameters:
#         c_field (numpy.ndarray): Input complex field.
#
#     Returns:
#         energy (float): Total energy of the field.
#     """
#     return np.sum(np.abs(c_field) ** 2)
#
#
# def update_plot(holo_field, focal_field, iteration, fig, axes):
#     """
#     Updates the real-time plots of the hologram plane and focal field phase.
#
#     Parameters:
#         holo_field (numpy.ndarray): Hologram field.
#         focal_field (numpy.ndarray): Focal field.
#         iteration (int): Current iteration.
#         fig (matplotlib.figure.Figure): Figure object.
#         axes (list): List of axes objects.
#     """
#     fig.suptitle(f'Iteration {iteration}')
#     axes[0].cla()
#     axes[0].imshow(np.angle(holo_field), cmap='twilight')
#     axes[0].set_title("Hologram Plane Phase")
#     axes[1].cla()
#     axes[1].imshow(np.abs(focal_field) ** 2, cmap='gray')
#     axes[1].set_title("Focal Field Intensity")
#     plt.pause(0.001)
#
#
# def create_noise_area(target_im, noise_area_factor):
#     """
#     Creates a modified target image with a noise area for phase retrieval.
#
#     Parameters:
#         target_im (numpy.ndarray): Input image.
#         noise_area_factor (float): Scaling factor for noise area size.
#
#     Returns:
#         im_with_noise_area (numpy.ndarray): Modified image with noise area.
#         mask_signal_area (numpy.ndarray): Signal area mask.
#         downscaled_im (numpy.ndarray): Downscaled version of the input image.
#     """
#     original_im, original_shape = target_im[:, :], target_im.shape
#     downscaled_im = rescale(original_im, 1 - noise_area_factor)
#     im_with_noise_area = pad2size(downscaled_im, *original_shape)
#     im_with_noise_area /= im_with_noise_area.max()
#     mask_signal_area = pad2size(np.ones_like(downscaled_im, dtype=bool), *original_shape)
#     return im_with_noise_area, mask_signal_area, downscaled_im
#
#
# def compute_gerchberg_saxton_hologram(target_im, initial_phase, n_iterations,
#                                       holo_padded_size_factor, use_noise_area, noise_area_factor):
#     """
#     Computes a phase-only hologram using the iterative Gerchberg-Saxton algorithm.
#
#     The algorithm alternates between the hologram plane and the focal plane, applying
#     constraints to refine the phase distribution iteratively.
#
#     Parameters:
#         target_im (numpy.ndarray): Target intensity image.
#         initial_phase (str): Type of initial phase ('Random' or 'Quadratic').
#         n_iterations (int): Number of algorithm iterations.
#         holo_padded_size_factor (float): Scaling factor for hologram size.
#         use_noise_area (bool): Whether to use a noise area.
#         noise_area_factor (float): Noise area factor.
#
#     Returns:
#         c_holo_eff (numpy.ndarray): Computed phase-only hologram field.
#     """
#     holo_shape = target_im.shape
#     target_im = rescale(target_im, holo_padded_size_factor)
#     mask_holo = create_square_mask(target_im.shape, holo_shape)
#     target_im, mask_signal_area, downscaled_im = create_noise_area(target_im, noise_area_factor)
#     target_ampl = np.sqrt(target_im)
#
#     # Initialize phase distribution
#     match initial_phase:
#         case 'Random':
#             phase0 = 2 * np.pi * np.random.rand(*target_ampl.shape)
#         case 'Quadratic':
#             x = np.arange(target_ampl.shape[1]) - target_ampl.shape[1] // 2
#             y = np.arange(target_ampl.shape[0]) - target_ampl.shape[0] // 2
#             X, Y = np.meshgrid(x, y)
#             k = np.pi / (1 * 2 * downscaled_im.shape[0])
#             phase0 = k * X ** 2 + k * Y ** 2
#         case _:
#             print('Unknown initial phase, using random initial phase')
#             phase0 = 2 * np.pi * np.random.rand(*target_ampl.shape)
#
#     plt.ion()
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
#     # Initialize focal plane field
#     c_focal_field = target_ampl * np.exp(1j * phase0)
#     complete_energy = compute_total_energy(c_focal_field)
#     for iteration in range(n_iterations):
#         # Back propagation to hologram plane
#         c_hologram_field = good_ifft2(c_focal_field, True)
#
#         # Apply phase-only constraint ensuring energy conservation and bandwidth constraint
#         c_hologram_field = np.exp(1j * np.angle(c_hologram_field))
#         c_hologram_field[~mask_holo] = 0
#         c_hologram_field *= np.sqrt(complete_energy / compute_total_energy(c_hologram_field))
#
#         # Forward propagation to focal plane
#         c_focal_field = good_fft2(c_hologram_field, True)
#
#         # Plot iterations
#         update_plot(c_hologram_field, c_focal_field, iteration + 1, fig, axes)
#
#         # Apply amplitude constraint
#         if not use_noise_area:
#             c_focal_field = target_ampl * np.exp(1j * np.angle(c_focal_field))
#             # c_focal_field *= np.sqrt(complete_energy / compute_total_energy(c_focal_field))
#         if use_noise_area:
#             c_focal_field[mask_signal_area] = np.sqrt(downscaled_im).flatten() * np.exp(1j * np.angle(c_focal_field[mask_signal_area]))
#             ed
#             # c_focal_field[mask_signal_area] *= np.sqrt((complete_energy - compute_total_energy(c_focal_field[~mask_signal_area])) / compute_total_energy(c_focal_field[mask_signal_area]))
#
#     return c_hologram_field[mask_holo].reshape(holo_shape)
#
#
# if __name__ == '__main__':
#     target_im = data.camera().astype(np.float64)
#     c_holo_eff = compute_gerchberg_saxton_hologram(target_im, 'Quadratic', 75,
#                                                    2, True, 0.2)
