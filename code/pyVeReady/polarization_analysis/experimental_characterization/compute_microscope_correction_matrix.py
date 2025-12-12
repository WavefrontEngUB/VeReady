import matplotlib
matplotlib.use('TkAgg')  # instead of Qt
from pyVeReady.polarization_analysis.polarization_analysis_functions import *
from pyVeReady.utils.paths_utils import *
from pyVeReady.utils.image_utils import *

if __name__ == "__main__":
    characterization_method = 'Rotating QWP'

    # Load Files
    x_polarization_tiff_filepath = ask_files_location('Select X Tiff File', True)
    y_polarization_tiff_filepath = ask_files_location('Select Y Tif File', True)

    x_data_list = load_tiff_images_to_list(x_polarization_tiff_filepath, slicing_index=0)
    y_data_list = load_tiff_images_to_list(y_polarization_tiff_filepath, slicing_index=0)

    # Compute Stokes
    s0_data_x, s1_data_x, s2_data_x, s3_data_x = compute_stokes(x_data_list, characterization_method)
    s0_data_y, s1_data_y, s2_data_y, s3_data_y = compute_stokes(y_data_list, characterization_method)

    # Perform Average Analysis
    intensity_threshold_x = s0_data_x.min() + 0.5 * s0_data_x.max() - s0_data_x.min()
    intensity_threshold_y = s0_data_y.min() + 0.5 * s0_data_y.max() - s0_data_y.min()

    averaged_data_x = [np.mean(data_im[s0_data_x > intensity_threshold_x]) for data_im in x_data_list]
    averaged_data_y = [np.mean(data_im[s0_data_y > intensity_threshold_y]) for data_im in y_data_list]

    s0_av_x, s1_av_x, s2_av_x, s3_av_x = compute_stokes(averaged_data_x, characterization_method)
    s0_av_y, s1_av_y, s2_av_y, s3_av_y = compute_stokes(averaged_data_y, characterization_method)

    # Show Polarization Ellipses
    maj_ax_x, min_ax_x, orientation_angle_x = compute_ellipse_pypol([s0_av_x, s1_av_x, s2_av_x, s3_av_x],
                                                                    parameter_type='Stokes',
                                                                    show=True)
    maj_ax_y, min_ax_y, orientation_angle_y = compute_ellipse_pypol([s0_av_y, s1_av_y, s2_av_y, s3_av_y],
                                                                    parameter_type='Stokes',
                                                                    show=True)

    ex_horizontal, ey_horizontal = stokes2jones_pypol(s0_av_x, s1_av_x, s2_av_x, s3_av_x)
    ex_vertical, ey_vertical = stokes2jones_pypol(s0_av_y, s1_av_y, s2_av_y, s3_av_y)

    norm_horizontal = np.sqrt(ex_horizontal**2 + ey_horizontal**2)
    norm_vertical = np.sqrt(ex_vertical**2 + ey_vertical**2)

    ex_horizontal_norm = ex_horizontal / norm_horizontal
    ey_horizontal_norm = ey_horizontal / norm_horizontal

    ex_vertical_norm = ex_vertical / norm_vertical
    ey_vertical_norm = ey_vertical / norm_vertical

    # Direct Matrix
    M = np.array([[ex_horizontal_norm, ex_vertical_norm],
                  [ey_horizontal_norm, ey_vertical_norm]])

    # Correction Matrix
    m_correction = np.linalg.inv(M)
    print('Correction Matrix')
    print(m_correction)
    print('A CORRECTIVE EXTRA PHASE IS STILL NEEDED, TO ESTIMATE IT MEASURE THE PURE X MODULATION WITH A'
          'FINE TUNE PHASE OF 0 AND CHARACTERIZE THE STOKES VECTORS, FROM THE AVERAGE ANALYSIS THE RELATIVE'
          'PHASE BETWEEN THE COMPONENTS IS THE APPROXIMATE CORRECTIVE PHASE')