from pyVeReady.utils.holography_utils import *
from pyVeReady.utils.image_utils import *


def normalize_array_min_max(arr):
    """
    Normalize a NumPy array to the range [0, 1].

    Parameters:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Normalized array with values scaled to [0, 1].
    """
    arr = np.asarray(arr, dtype=np.float64)
    amin, amax = arr.min(), arr.max()
    if amax == amin:  # avoid division by zero
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)


def generate_multipoint_target(shape, periodicity, n_points):
    """
    Creates a 2D square zero matrix with a centered grid of focal spots.

    Parameters:
        shape (tuple of int): Shape of the output array (must be 2D and square).
        periodicity (int): Distance between spots in pixels.
        n_points (int): Number of spots along one axis (total spots = n_points^2).

    Returns:
        numpy.ndarray: 2D array with ones placed at spot positions.
    """
    # enforce square 2D shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Shape must be 2D and square.")

    n = shape[0]
    arr = np.zeros(shape, dtype=np.float64)

    center = n // 2

    # generate coordinates for a centered grid
    half = (n_points - 1) // 2
    offsets = np.arange(-half, half + 1) * periodicity

    for dy in offsets:
        for dx in offsets:
            x, y = center + dx, center + dy
            if 0 <= x < n and 0 <= y < n:
                arr[y, x] = 1

    return arr


def weighted_gerchberg_saxton(target_intensity, initial_phase, n_iter=30, fix_phase_from=30):
    # Get a mask of the focal_spots
    mask_focal_spots = target_intensity != 0

    # Initialize initial phase
    phase_slm = initial_phase

    # Initialize Complex Electric Fields
    holo_field = np.zeros(target_intensity.shape, dtype=np.complex128)
    focal_field = np.zeros(target_intensity.shape, dtype=np.complex128)

    # Initialize Beam Source and Weights
    beam = np.ones(target_intensity.shape, dtype=np.float64)
    weights = np.ones(target_intensity.shape, dtype=np.float64)

    # Initialize Performance Indicators
    performances = []

    # GS Iterations
    for n in range(n_iter):
        # Impose Beam profile (Phase-only modulation constraint)
        holo_field = beam * np.exp(1j * phase_slm)

        # Forward Propagate to Focal Plane
        focal_field = centered_fft2(holo_field)

        # Update phase at focal plane if desired, if not keep phase at focal plane fixed
        # This phase does not correspond to the SLM hologram's phase
        if n < fix_phase_from:
            phase_focal = np.angle(focal_field)

        # Compute new Weights at the Focal Spots Position
        reconstructed_intensity = np.abs(focal_field) ** 2
        normalized_intensity = (reconstructed_intensity - reconstructed_intensity.min()) / (reconstructed_intensity.max() - reconstructed_intensity.min())
        weights[mask_focal_spots] = np.sqrt(target_intensity[mask_focal_spots] / normalized_intensity[mask_focal_spots]) * weights[mask_focal_spots]

        # Impose Target Amplitude Constraint with weights
        focal_field = weights * np.sqrt(target_intensity) * np.exp(1j * phase_focal)

        # Back Propagate to SLM plane
        holo_field = centered_ifft2(focal_field)
        phase_slm = np.angle(holo_field)

    return phase_slm


if __name__ == '__main__':
    mm, um, nm = 1e-3, 1e-6, 1e-9
    np.random.seed(777)

    # Set WGS Parameters
    holo_shape = (512, 512)
    n_points = 21
    periodicity = 3
    target = generate_multipoint_target(holo_shape, periodicity, n_points)
    initial_phase = np.random.rand(*target.shape) * 2 * np.pi
    multi_point_hologram = weighted_gerchberg_saxton(target, initial_phase, n_iter=100, fix_phase_from=1e3)

    # Compute Spacing and FOV physical Sizes
    microscope_magnification = 106
    resolution_at_sample_plane = compute_image_plane_resolution(wavelength=488 * nm,
                                                                slm_pixel_pitch=12.5 * um,
                                                                hologram_pixels=holo_shape[0]) / microscope_magnification
    print(f"Spot Spacing at Sample Plane: {resolution_at_sample_plane * periodicity / um:.2f} um")
    print(f"Imaging Field of View at Sample Plane: {resolution_at_sample_plane * periodicity * n_points / um:.2f} um")

    # Simulate Reconstruction
    beam = np.ones(multi_point_hologram.shape, dtype=np.float64)
    slm_field_multipoint = beam * np.exp(1j * multi_point_hologram)
    reconstructed_target = np.abs(centered_fft2(slm_field_multipoint)) ** 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].imshow(target, cmap="hot")
    axes[0].set_title("Target")
    axes[1].imshow(multi_point_hologram, cmap="gray")
    axes[1].set_title("Hologram")
    axes[2].imshow(reconstructed_target, cmap="hot")
    axes[2].set_title("Reconstructed")
    plt.tight_layout()

    # Simulate Reconstruction with Focal Plane Upsampling
    topological_charge = 1
    X, Y, _, _ = pixel_coordinates(multi_point_hologram)
    beam_circ = np.float64((X**2 + Y**2) <= X.max()**2)
    vortex_phase = np.arctan2(Y, X) * topological_charge

    slm_field_simulation = beam_circ * np.exp(1j * multi_point_hologram) * np.exp(1j * vortex_phase)
    slm_field_simulation[np.abs(slm_field_simulation) == 0] = 0

    n_times_upsampling = 8
    slm_field_simulation_up = pad2size(slm_field_simulation,
                                       slm_field_simulation.shape[0] * n_times_upsampling,
                                       slm_field_simulation.shape[1] * n_times_upsampling)

    reconstructed_multi_donuts = np.abs(centered_fft2(slm_field_simulation_up)) ** 2
    reconstructed_multi_donuts = normalize_array_min_max(reconstructed_multi_donuts)
    slm_phase_modulation = (np.angle(slm_field_simulation_up) + 2*np.pi) % (2*np.pi)

    fig_up, axes_up = plt.subplots(1, 2)
    axes_up[0].imshow(slm_phase_modulation, cmap="gray")
    axes_up[0].set_title("Multi Donut Phase Modulation")
    axes_up[1].imshow(reconstructed_multi_donuts, cmap="gray")
    axes_up[1].set_title("Reconstructed")
    plt.tight_layout()

    plt.show()
