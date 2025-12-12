import numpy as np
from pyVeReady.utils.holography_utils import *
from pyVeReady.holo_generation.holo_generation_functions import *
import slmpy
np.random.seed(1)
from scipy.signal import fftconvolve
from tqdm import tqdm

def generate_points(size, spacing, span):
    """
    Generates a square 2D array with ones evenly spaced in both directions within a square span window.

    Parameters:
        size (int): The size (size x size) of the square array.
        spacing (int): The spacing between ones.
        span (int): The size of the square region where ones are placed.

    Returns:
        np.ndarray: A (size x size) array with ones placed in a grid pattern within the span window.
    """
    # Initialize a zero matrix
    array = np.zeros((size, size), dtype=int)

    # Compute the center coordinates of the array
    center = size // 2
    half_span = span // 2

    # Define the start and end indices for the span window
    x_start, x_end = center - half_span, center + half_span
    y_start, y_end = center - half_span, center + half_span

    # Place ones at evenly spaced positions inside the span window
    for i in range(x_start, x_end + 1, spacing):
        for j in range(y_start, y_end + 1, spacing):
            if 0 <= i < size and 0 <= j < size:  # Ensure indices are within bounds
                array[i, j] = 1

    return array


def update_plot(hologram_field, focal_field, iteration, n_iterations):
    """Computes phase and intensity, then updates the real-time plot."""
    hologram_phase = np.angle(hologram_field)  # Compute phase of the hologram
    focal_intensity = np.abs(focal_field) ** 2  # Compute intensity of the focal field

    plt.clf()  # Clear figure

    # Create subplots
    plt.subplot(1, 2, 1)
    plt.imshow(hologram_phase, cmap='twilight', extent=[-1, 1, -1, 1])
    plt.colorbar()
    plt.title(f'Hologram Phase (Iteration {iteration}/{n_iterations})')

    plt.subplot(1, 2, 2)
    plt.imshow(focal_intensity, cmap='inferno', extent=[-1, 1, -1, 1])
    plt.colorbar()
    plt.title(f'Focal Field Intensity')

    plt.pause(0.001)  # Pause to allow real-time update
    plt.show()


# Define Parameters
holo_shape = (460, 460)
beam_diameter_mm = 5.75
diam_pix_beam = int(beam_diameter_mm*1e3 / 12.5)

padding_factor = 1
n_iterations = 60

# Load target intensity, DELTAS pattern, already has the padded hologram shape.
target_positions = generate_points(holo_shape[0] * padding_factor, 12 * padding_factor, 14 * padding_factor)

# Compute the diffraction limited donut
X_h, Y_h, R_h, phi_h = pixel_coordinates(target_positions)
pupil_mask = (X_h**2 + Y_h**2) <= (diam_pix_beam/2)**2
c_holo_donut = np.exp(1j * phi_h) * pupil_mask
c_holo_donut[np.abs(c_holo_donut) == 0] = 0
c_focal_donut = centered_fft2(c_holo_donut)
ampl_donut = np.abs(c_focal_donut)

effective_holo_mask = create_square_mask(target_positions.shape, holo_shape)

# Compute target amplitude
target_amplitude = np.abs(fftconvolve(target_positions, ampl_donut, 'same'))
plt.figure()
plt.title('Target Intensity')
plt.imshow(target_amplitude**2)

# Gerchberg-Saxton Algorithm
initial_phase_type = 'Random'

match initial_phase_type:
    case 'Random':
        phase0 = 2 * np.pi * np.random.rand(*target_amplitude.shape)
    case 'Quadratic':
        x = np.arange(target_amplitude.shape[1]) - target_amplitude.shape[1] // 2
        y = np.arange(target_amplitude.shape[0]) - target_amplitude.shape[0] // 2
        X, Y = np.meshgrid(x, y)
        k = np.pi / (2 * target_amplitude.shape[0])
        phase0 = k * X ** 2 + k * Y ** 2
    case _:
        print('Unknown initial phase, random initial phase generated')
        phase0 = 2 * np.pi * np.random.rand(*target_amplitude.shape)

focal_field = target_amplitude * np.exp(1j * phase0)
hologram_field = None

# Enable interactive plotting
plt.figure(figsize=(10, 5))
plt.ion()
for ii in tqdm(range(n_iterations), desc="GS Iterating"):
    hologram_field = centered_ifft2(focal_field)  # Propagate backwards to hologram plane
    hologram_field = np.exp(1j * np.angle(hologram_field))  # Apply phase-only constraint
    hologram_field = hologram_field * pupil_mask
    hologram_field[np.abs(hologram_field) == 0] = 0

    focal_field = centered_fft2(hologram_field)  # Propagate forward to focal plane
    update_plot(hologram_field, focal_field, ii + 1, n_iterations)  # Update real-time visualization

    if ii != n_iterations - 1:  # Apply amplitude constraint
        focal_field = target_amplitude * np.exp(1j * np.angle(focal_field))

c_hologram_effective = hologram_field[effective_holo_mask].reshape(holo_shape)

# Add vortex phase and linear phase
X_c, Y_c, R_c, phi_c = pixel_coordinates(np.ones_like(c_hologram_effective))
holo_pupil_c = (X_c ** 2 + Y_c ** 2) <= (diam_pix_beam / 2) ** 2  # Beam diameter of 5.75mm
c_hologram_effective *= holo_pupil_c * np.exp(1j * 2 * np.pi * X_c / 14.5)
c_hologram_effective[np.abs(c_hologram_effective == 0)] = 0

# Send to SLM
row_semi, cols_semi = 1024, 640
slm_lut = SLMLut(0, 195)
hologram_u8 = c_phase2gray_interp(c_hologram_effective, slm_lut.gray_level_lut, slm_lut.phase_lut)

hologram_semi_display = pad2size(hologram_u8, row_semi, cols_semi)
empty_semi_display = np.zeros((row_semi, cols_semi), dtype=np.uint8)
slm_display = np.hstack((empty_semi_display, hologram_semi_display)).astype(np.uint8)

plt.figure()
plt.imshow(slm_display, cmap='gray')

slm = slmpy.SLMdisplay(monitor=1)
slm.updateArray(slm_display)