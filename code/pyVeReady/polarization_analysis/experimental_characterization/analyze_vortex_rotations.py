from pyVeReady.polarization_analysis.polarization_analysis_functions import *
from pyVeReady.utils.image_utils import *


# Experiment Parameters
vortex_rotation_step = 1  # [Degrees]
n_rotations_qwp = 8
show_fitting = False

# Load Data
array_all_measures = np.load('20250128_vortex_rotations.npy')
list_all_measures = [array_all_measures[:,:,ii] for ii in range(array_all_measures.shape[2])]
list_of_grouped_measures = [list_all_measures[jj:jj + n_rotations_qwp] for jj in range(0, array_all_measures.shape[2], n_rotations_qwp)]

# data_list = list_of_grouped_measures[0]
list_of_grouped_measures = list_of_grouped_measures[:]
orientation_mean_error = np.zeros(len(list_of_grouped_measures), dtype = np.float64)
for measure_idx, data_list in enumerate(list_of_grouped_measures):

    # Compute Stokes Parameters and Ellipse parameters
    s0, s1, s2, s3 = compute_stokes_rotating_qwp(data_list)
    maj_ax, min_ax, orientation_map = compute_ellipse_pypol([s0, s1, s2, s3],
                                                            parameter_type = 'Stokes')
    error_map, mean_absolute_error = compute_azimuthal_angle_error_map(orientation_map, s0,
                                                                       show_center_estimation = False)
    orientation_mean_error[measure_idx] = mean_absolute_error

    print(f'Measure {measure_idx}. Mean absolute error: {mean_absolute_error * 180/np.pi:.2f} Degrees')

best_measure_idx = np.argmin(np.abs(orientation_mean_error))
print(f'Best Measure {best_measure_idx} with Offset Angle: {orientation_mean_error[best_measure_idx] * 180/np.pi:.3} degrees')

s0_best, s1_best, s2_best, s3_best = compute_stokes_rotating_qwp(list_of_grouped_measures[best_measure_idx])
show_stokes_map(s0_best, s1_best, s2_best, s3_best)
(fig, axes,
 Ex_data, Ey_data,
 relative_phase,
 orientation_map_first_quadrant,
 error) = polarization_map_analysis([s0_best, s1_best, s2_best, s3_best],
                                    vector_tpye = 'Stokes',
                                    show_figure = True,
                                    show_relative_phase_map = True,
                                    show_azimuthal_orientation_map= True,
                                    orientation_into_first_quadrant = True,
                                    draw_step = 10,
                                    figname = 'Best Azimuthal Beam')
