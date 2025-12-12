import slmpy

from pyVeReady.holo_generation.holo_generation_functions import *
from pyVeReady.utils.holography_utils import *
from skimage import data
from skimage.transform import rescale
from pyVeReady.utils.image_utils import *
mm, um, nm = 1e-3, 1e-6, 1e-9

def update_hologram_plots(beam_profile, hologram_field, iteration, axes, imshow_objects):
    """Update beam profile, hologram amplitude, and phase without resetting zoom."""
    hologram_ampl = np.abs(hologram_field)
    hologram_phase = np.angle(hologram_field)

    # Update image data instead of clearing the axes
    imshow_objects[0].set_data(beam_profile)
    imshow_objects[1].set_data(hologram_ampl)
    imshow_objects[2].set_data(hologram_phase)

    # Adjust vmax for amplitude
    imshow_objects[1].set_clim(vmin=0, vmax=hologram_ampl.max())

    # Update titles
    axes[0, 0].set_title('Beam Profile')
    axes[0, 1].set_title(f'Hologram Amplitude (Iteration {iteration})')
    axes[0, 2].set_title(f'Hologram Phase (Iteration {iteration})')

    plt.pause(0.001)

def update_focal_plots(target_intensity, focal_field, iteration, axes, imshow_objects):
    """Update target intensity, focal field intensity, and phase without resetting zoom."""
    focal_intensity = np.abs(focal_field) ** 2
    focal_phase = np.angle(focal_field)

    # Update image data instead of clearing the axes
    imshow_objects[3].set_data(target_intensity)
    imshow_objects[4].set_data(focal_intensity)
    imshow_objects[5].set_data(focal_phase)

    # Adjust vmax for intensity
    imshow_objects[4].set_clim(vmin=0, vmax=focal_intensity.max())

    # Update titles
    axes[1, 0].set_title('Target Intensity')
    axes[1, 1].set_title(f'Focal Field Intensity (Iteration {iteration})')
    axes[1, 2].set_title(f'Focal Field Phase (Iteration {iteration})')

    plt.pause(0.001)

# Set up seed
np.random.seed(27)

# Set up GS parameters
holo_shape = (640, 640)
n_iter = 100

# Load and preprocess target intensity
target_intensity = data.camera()
target_intensity = rescale(target_intensity, 0.25)
target_intensity = pad2size(target_intensity, holo_shape[0], holo_shape[1])
target_intensity = generate_points(holo_shape[0], 15, 60, 7, 34)
# target_intensity = interactive_point_selection(holo_shape[0])
target_intensity = target_intensity / target_intensity.max()
target_ampl = np.sqrt(target_intensity)

# Initialize hologram
phase_0 = compute_initial_phase(target_intensity.shape, 'Random')
beam_profile = np.ones_like(target_intensity)
beam_profile *= np.sqrt(compute_total_energy(target_intensity) / compute_total_energy(beam_profile))  # Impose same energy
hologram_field = beam_profile * np.exp(1j * phase_0)

# Create figure and subplots with shared x/y axes per row and initial image objects for real time plot management
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex='row', sharey='row')
imshow_objects = [
    axes[0, 0].imshow(beam_profile, cmap='hot', vmin=0),
    axes[0, 1].imshow(np.abs(hologram_field), cmap='hot', vmin=0),
    axes[0, 2].imshow(np.angle(hologram_field), cmap='twilight'),
    axes[1, 0].imshow(target_intensity, cmap='hot', vmin=0),
    axes[1, 1].imshow(np.abs(hologram_field) ** 2, cmap='hot', vmin=0),
    axes[1, 2].imshow(np.angle(hologram_field), cmap='twilight')
]

# GS Algorithm
for ii in range(n_iter):
    focal_field = centered_fft2(hologram_field, True)
    update_focal_plots(target_intensity, focal_field, ii + 1, axes, imshow_objects)

    if ii != n_iter - 1:
        # focal_field = target_ampl * np.exp(1j * np.angle(focal_field))
        w = np.exp((target_ampl - np.abs(focal_field)))
        w = 1
        focal_field = target_ampl * w * np.exp(1j * np.angle(focal_field))
        hologram_field = centered_ifft2(focal_field, True)
        update_hologram_plots(beam_profile, hologram_field, ii + 1, axes, imshow_objects)
        hologram_field = beam_profile * np.exp(1j * np.angle(hologram_field))

# Add extra phase terms
add_vortex_phase, m = True, 1
add_linear_phase = False
add_pupil_mask, beam_diam_mm = False, 5.75
add_correction_map_double_pass, slm_correction_filepath = True, r'C:\Users\Usuario\OneDrive - Universitat de Barcelona\PhD\VeReady\Code\Vector_Beams_LabVIEW\CAL_LSH0804783_490nm.bmp'

X_h, Y_h, R_h, phi_h = pixel_coordinates(hologram_field)
if add_vortex_phase:
    hologram_field = hologram_field * np.exp(1j * m * phi_h)
if add_linear_phase:
    hologram_field = hologram_field * np.exp(1j * 2*np.pi * X_h/20.5)
if add_pupil_mask:
    pupil_mask = (X_h**2 + Y_h**2) <= ((beam_diam_mm * 1e3 / 12.5) / 2)**2
    hologram_field = hologram_field * pupil_mask
    hologram_field[np.abs(hologram_field) == 0] = 0

slm_sub_display_shape = (1024, 640)
slm_shape = (1024, 1280)
slm_pixel_size = 12.5 * um
focusing_lens_focal = 300 * mm
laser_wavelength = 488 * nm
defocus_mm = 10
slm_lut = SLMLut(0, 195)

c_display_b = pad2size(hologram_field, slm_sub_display_shape[0], slm_sub_display_shape[1])
c_display_a = np.zeros_like(c_display_b)
c_display_slm = np.hstack((c_display_a, c_display_b))

X_slm, Y_slm, R_slm, phi_slm = pixel_coordinates(c_display_slm)
if add_correction_map_double_pass:
    correction_map_phase = generate_double_pass_correction_map(slm_correction_filepath, 1024, 1280,
                                                               195, 2000,
                                                               3, 3, 0, 0)
    c_display_slm = c_display_slm * np.exp(1j * correction_map_phase)

    # Apply quadratic phase to correct focus if camera is placed at the focal plane of the uncorrected beam
    c_defocus = np.exp(1j * np.pi * (R_slm * slm_pixel_size) ** 2 * (1 / (focusing_lens_focal - defocus_mm * mm) - 1 / focusing_lens_focal) / laser_wavelength)
    c_display_slm = c_display_slm * c_defocus

c_display_slm[np.abs(c_display_slm) == 0] = 0
display_slm_u8 = c_phase2gray_interp(c_display_slm, slm_lut.gray_level_lut, slm_lut.phase_lut).astype(np.uint8)

slm = slmpy.SLMdisplay()
slm.updateArray(display_slm_u8)
