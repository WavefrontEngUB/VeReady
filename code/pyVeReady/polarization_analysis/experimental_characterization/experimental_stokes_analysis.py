import matplotlib
matplotlib.use('TkAgg')  # instead of Qt
from scipy.optimize import curve_fit
from pyVeReady.polarization_analysis.polarization_analysis_functions import *
from pyVeReady.utils.paths_utils import *
from pyVeReady.utils.image_utils import *
import os
import glob


average_analysis = True
fit_azimuthal = False
characterization_method = 'Rotating QWP'
examination_plane = 'Pupil'  # 'Pupil' or 'Focal' planes
filter_arrow_plotting_by_intensity = False


# Load Files
tiff_filepaths = ask_files_location('Select Tiff Files')
if len(tiff_filepaths) > 1:
    tiff_filepaths = sorted(tiff_filepaths, key=lambda filepath: int(os.path.splitext(os.path.basename(filepath))[0]))

data_list = load_tiff_images_to_list(tiff_filepaths, slicing_index=0)
print(f'Loaded File:\n{tiff_filepaths[0]}')
print(f'Stokes Characterization Method: {characterization_method}')

# data_centered = center_image_to_reference(data_list, data_list[0])
# data_centered = [crop_array_centered(im, 145, 300, (400, 400)) for im in data_centered]

data_centered = data_list
if examination_plane == 'Focal':
    roi_size = 32
    row, col = find_beam_center(np.max(np.array(data_list), axis=0), '')
    data_centered = [crop_image(im, row, col, roi_size) for im in data_list]

s0_data, s1_data, s2_data, s3_data = compute_stokes(data_centered, characterization_method)
show_stokes_map(s0_data, s1_data, s2_data, s3_data)

arrow_drawing_step = 1 if examination_plane == 'Focal' else 10

(fig_pol, ax_pol,
 Ex, Ey, relative_phase,
 orientation_map,
 error) = polarization_map_analysis([s0_data, s1_data, s2_data, s3_data],
                                    vector_tpye='Stokes',
                                    show_figure=True,
                                    show_relative_phase_map=False,
                                    show_azimuthal_orientation_map=False,
                                    orientation_into_first_quadrant=False,
                                    draw_step=arrow_drawing_step,
                                    filter_arrow_intensity=filter_arrow_plotting_by_intensity)


if fit_azimuthal:
    error_map, mean_absolute_error = compute_azimuthal_angle_error_map(orientation_map, s0_data, roi_search=25, show_center_estimation=True)
    print(f'Mean Absolute Error: {mean_absolute_error * 180 / np.pi:.3f} Degrees')

if average_analysis:
    intensity_threshold = s0_data.min() + 0.2 * (s0_data.max() - s0_data.min())
    averaged_data = [np.mean(data_im[s0_data > intensity_threshold]) for data_im in data_centered]
    s0_av, s1_av, s2_av, s3_av = compute_stokes(averaged_data, characterization_method)

    maj_ax, min_ax, orientation_angle = compute_ellipse_pypol([s0_av, s1_av, s2_av, s3_av],
                                                                    parameter_type='Stokes',
                                                                    show=True)


    print(f'Averaged Stokes Parameters')
    print(f'S0: {s0_av:.3f}')
    print(f'S1: {s1_av:.3f}')
    print(f'S2: {s2_av:.3f}')
    print(f'S3: {s3_av:.3f}')

