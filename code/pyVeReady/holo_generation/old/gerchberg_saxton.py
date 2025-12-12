from pyVeReady.utils.holography_utils import *
from pyVeReady.holo_generation.old.generate_ub_points import *
from skimage import data
from skimage.transform import rescale
from pyVeReady.utils.image_utils import *
matplotlib.use('tkagg')
np.random.seed(1)


def update_plot(holo_field, focal_field, iteration, fig, axes, adjust_vmax=False):
    """
    Updates the real-time plots of the hologram plane and focal field phase.

    Parameters:
    - adjust_vmax (bool): If True, adjust the vmax for the focal field intensity to better show the signal.
    """
    fig.suptitle(f'Iteration {iteration}')

    # Update hologram plane phase
    axes[0].cla()
    axes[0].imshow(np.angle(holo_field), cmap='twilight')
    axes[0].set_title("Hologram Plane Phase")

    # Update focal field intensity
    axes[1].cla()

    if adjust_vmax:
        # Compute vmax based on the signal and noise
        intensity = np.abs(focal_field) ** 2

        # Compute the 95th percentile as an appropriate cutoff for vmax
        vmax = np.percentile(intensity, 98.5)  # Using 98.5th percentile to set vmax

        # Update the plot with the new vmax value
        axes[1].imshow(intensity, cmap='gray', vmax=vmax)
    else:
        axes[1].imshow(np.abs(focal_field) ** 2, cmap='gray')

    axes[1].set_title("Focal Field Intensity")

    # Pause to update the plot
    plt.pause(0.001)


def compute_gerchberg_saxton_hologram(target_im, initial_phase, n_iterations,
                                      holo_padded_size_factor, use_noise_area, noise_area_factor):
    # Define the shape of the effective hologram, since the hologram field might be zero padded
    effective_holo_shape = target_im.shape

    # Upsample target image if hologram is zero padded (to obtain higher resolution in the focal plane)
    target_im = rescale(target_im, holo_padded_size_factor)
    mask_holo = create_square_mask(target_im.shape, effective_holo_shape)

    # Create Noise Area in the target image if desired
    target_im, mask_signal_area, downscaled_im = create_noise_area(target_im, noise_area_factor)

    # Compute Amplitude and Initial Phase of the electric field at the Focal Plane
    target_ampl = np.sqrt(target_im)
    match initial_phase:
        case 'Random':
            phase0 = 2 * np.pi * np.random.rand(*target_ampl.shape)
        case 'Quadratic':
            x = np.arange(target_ampl.shape[1]) - target_ampl.shape[1] // 2
            y = np.arange(target_ampl.shape[0]) - target_ampl.shape[0] // 2
            X, Y = np.meshgrid(x, y)
            k = np.pi / (0.975 * 2 * target_ampl.shape[0])
            phase0 = k * X**2 + k * Y**2
        case _:
            print('Unknown initial phase, random initial phase generated')
            phase0 = 2 * np.pi * np.random.rand(*target_ampl.shape)

    fig_gs, axes_gs = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()  # Turn on interactive mode for real-time updates

    # Initialize Field at Focal Plane
    c_focal_field = target_ampl * np.exp(1j * phase0)
    c_hologram_field = None  # Just so PyCharm doesn't complain
    total_energy = compute_total_energy(c_focal_field)
    for iteration in range(n_iterations):
        # Back propagation to hologram plane
        c_hologram_field = centered_ifft2(c_focal_field, energy_preservation=True)

        # Apply phase-only constraint ensuring energy conservation and bandwidth constraint
        c_hologram_field = np.exp(1j * np.angle(c_hologram_field))
        c_hologram_field[~mask_holo] = 0
        c_hologram_field *= np.sqrt(total_energy / compute_total_energy(c_hologram_field))

        # Forward propagation to focal plane
        c_focal_field = centered_fft2(c_hologram_field, energy_preservation=True)

        # Plot iterations
        update_plot(c_hologram_field, c_focal_field, iteration + 1, fig_gs, axes_gs, False)

        # Apply amplitude constraint except for the last iteration, to keep the reconstructed focal field
        if iteration != n_iterations-1:
            if not use_noise_area:
                # Amplitude Constraint in all focal plane
                c_focal_field = target_ampl * np.exp(1j * np.angle(c_focal_field))
                # c_focal_field *= np.sqrt(complete_energy / compute_total_energy(c_focal_field))
            if use_noise_area:
                # Amplitude constraint only at signal area
                c_focal_field[mask_signal_area] = np.sqrt(downscaled_im).flatten() * np.exp(1j * np.angle(c_focal_field[mask_signal_area]))
                # c_focal_field *= np.sqrt(complete_energy / compute_total_energy(c_focal_field))

                # Preserve energy leaving noise area unaffected
                c_focal_field[mask_signal_area] *= np.sqrt((total_energy - compute_total_energy(c_focal_field[~mask_signal_area])) / compute_total_energy(c_focal_field))

    c_holo_eff = c_hologram_field[mask_holo == 1].reshape(effective_holo_shape)
    return c_holo_eff


if __name__ == '__main__':
    test_selection = 0  # 0 for GS Algorithm test and 1 for Fourier Transforms Test
    if test_selection == 0:
        # Load Target Image
        target_img_test = data.camera().astype(np.float64)

        use_ub_points = True
        if use_ub_points:
            target_img_test = create_ub_image(480, 480, 90, row_spacing=7, col_spacing=7)  # Adjust row and column spacing here

        # Gerchberg-Saxton Parameters
        holo_padded_size_factor_test = 1
        noise_area_factor_test = 0
        initial_phase_test = 'Quadratic'
        use_noise_area_test = False
        n_iterations_test = 50

        c_holo_effective = compute_gerchberg_saxton_hologram(target_img_test, initial_phase_test, n_iterations_test,
                                                             holo_padded_size_factor_test, use_noise_area_test, noise_area_factor_test)

        if use_ub_points:
            n_pad = 4
            c_padded_holo = pad2size(c_holo_effective,
                                     n_pad*c_holo_effective.shape[0], n_pad*c_holo_effective.shape[1])
            X, Y, R, phi = pixel_coordinates(c_padded_holo)
            c_padded_holo *= np.exp(1j * phi)
            c_padded_holo[np.abs(c_padded_holo) == 0] = 0
            c_focal = centered_fft2(c_padded_holo)

            # Plot the phase of the hologram and the intensity of the focal field
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Phase of the hologram
            axs[0].imshow(np.angle(c_padded_holo), cmap='twilight', extent=[-1, 1, -1, 1])
            axs[0].set_title("Phase of Hologram")
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")
            axs[0].axis("image")

            # Intensity of the focal field
            axs[1].imshow(np.abs(c_focal) ** 2, cmap='gray', extent=[-1, 1, -1, 1])
            axs[1].set_title("Intensity of Focal Field")
            axs[1].set_xlabel("X")
            axs[1].set_ylabel("Y")
            axs[1].axis("image")

            plt.tight_layout()

    if test_selection == 1:
        # aperture_slit = centered_rectangle(800, 800, 50, 25)
        aperture_slit = centered_triangle(800, 800, 50, 100)
        X, Y, R, phi = pixel_coordinates(aperture_slit)
        complex_aperture = np.exp(1j * phi) * aperture_slit
        complex_aperture[np.abs(complex_aperture == 0)] = 0

        diffraction_pattern = centered_fft2(complex_aperture)

        input_diffraction_pattern = diffraction_pattern[:,:]
        recovered_complex_aperture = centered_ifft2(diffraction_pattern)

        # Plot results
        fig, axes = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(12, 6))

        # Original Aperture
        axes[0, 0].imshow(np.abs(complex_aperture), cmap='gray')
        axes[0, 0].set_title("Aperture")
        axes[0, 0].axis("off")

        # Phase of Aperture
        axes[0, 1].imshow(np.angle(complex_aperture), cmap='twilight')
        axes[0, 1].set_title("Phase of Aperture")
        axes[0, 1].axis("off")

        # Amplitude of Diffraction Pattern
        axes[0, 2].imshow(np.abs(diffraction_pattern), cmap='inferno')
        axes[0, 2].set_title("Amplitude of Diffraction")
        axes[0, 2].axis("off")

        # Phase of Diffraction Pattern
        axes[0, 3].imshow(np.angle(diffraction_pattern), cmap='twilight')
        axes[0, 3].set_title("Phase of Diffraction")
        axes[0, 3].axis("off")

        # Amplitude of Input Diffraction Pattern
        axes[1, 0].imshow(np.abs(input_diffraction_pattern), cmap='inferno')
        axes[1, 0].set_title("Amplitude of Input Diffraction")
        axes[1, 0].axis("off")

        # Phase of Input Diffraction Pattern
        axes[1, 1].imshow(np.angle(input_diffraction_pattern), cmap='twilight')
        axes[1, 1].set_title("Phase of Input Diffraction")
        axes[1, 1].axis("off")

        # Amplitude of Recovered Aperture
        axes[1, 2].imshow(np.abs(recovered_complex_aperture), cmap='gray')
        axes[1, 2].set_title("Amplitude of Recovered Aperture")
        axes[1, 2].axis("off")

        # Phase of Recovered Aperture
        axes[1, 3].imshow(np.angle(recovered_complex_aperture), cmap='twilight')
        axes[1, 3].set_title("Phase of Recovered Aperture")
        axes[1, 3].axis("off")

        plt.tight_layout()
        plt.show()
