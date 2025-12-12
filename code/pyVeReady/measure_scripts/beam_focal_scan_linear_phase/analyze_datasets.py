from pyVeReady.utils.paths_utils import *
from pyVeReady.polarization_analysis.polarization_analysis_functions import *
from pyVeReady.utils.image_utils import *
import numpy as np
from scipy.interpolate import RBFInterpolator
from tqdm import tqdm
import pickle
import matplotlib
matplotlib.use('Tkagg')
mm, um, nm = 1e-3, 1e-6, 1e-9


def compute_maximum_projection_stokes_from_scan(list_of_measures):
    stokes_params_all = [compute_stokes(measure, 'Rotating QWP') for measure in
                         tqdm(list_of_measures, desc="Computing Stokes Parameters")]
    s0_all, s1_all, s2_all, s3_all = (np.array(param, dtype=np.float64) for param in zip(*stokes_params_all))

    s0_max_proj = np.take_along_axis(s0_all, np.abs(s0_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)
    s1_max_proj = np.take_along_axis(s1_all, np.abs(s1_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)
    s2_max_proj = np.take_along_axis(s2_all, np.abs(s2_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)
    s3_max_proj = np.take_along_axis(s3_all, np.abs(s3_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)
    return s0_max_proj, s1_max_proj, s2_max_proj, s3_max_proj

def compute_polarization_correction_map(measurements, modulation_type, n_steps, eff_pixel_size, camera_pixel_size_mm,
                                        roi_size, periodsX_mesh, periodsY_mesh,
                                        stokes_display='Maximum Projection',
                                        return_max_projection_stokes=False):
    # Initialize correction phase map and define target relative phase
    correction_phase_map = np.zeros_like(periodsX_mesh, dtype=np.float64)
    target_relative_phase = {'Circular Right': np.pi / 2, 'Laguerre-Gaussian Right': np.pi / 2,
                             'Circular Left': -np.pi / 2, 'Laguerre-Gaussian Left': -np.pi / 2,
                             'Azimuthal Polarization': 0, 'Zeros Hologram': 0}.get(modulation_type, 0)

    # Compute Stokes Parameters and Jones Vectors for all measurements
    stokes_params_all = [compute_stokes(measure, 'Rotating QWP') for measure in
                         tqdm(measurements, desc="Computing Stokes Parameters")]
    s0_all, s1_all, s2_all, s3_all = (np.array(param, dtype=np.float64) for param in zip(*stokes_params_all))

    # Crop and calculate the correction phase map
    cropped_s0_list, cropped_s1_norm_list, cropped_s2_norm_list, cropped_s3_norm_list = [], [], [], []
    for slice_idx, (ii, jj) in enumerate(
            tqdm(np.ndindex(periodsX_mesh.shape), desc="Processing Polarization Correction Phase Map", total=periodsX_mesh.size)):
        s0, s1, s2, s3 = s0_all[slice_idx], s1_all[slice_idx], s2_all[slice_idx], s3_all[slice_idx]

        # Locate the Beam Center and crop a ROI around it
        row_c, col_c = find_beam_center(s0, thresholding='')
        cropped_s0 = crop_image(s0, row_c, col_c, roi_size)
        cropped_s1 = crop_image(s1, row_c, col_c, roi_size)
        cropped_s2 = crop_image(s2, row_c, col_c, roi_size)
        cropped_s3 = crop_image(s3, row_c, col_c, roi_size)
        cropped_s0_list.append(cropped_s0), cropped_s1_norm_list.append(cropped_s1/cropped_s0)
        cropped_s2_norm_list.append(cropped_s2/cropped_s0), cropped_s3_norm_list.append(cropped_s3/cropped_s0)

        cropped_ex, cropped_ey = stokes2jones_pypol(cropped_s0, cropped_s1, cropped_s2, cropped_s3)

        # Compute the Relative Phase for all the ROI (varies point by point)
        relative_phase_cropped = np.angle(np.exp(1j * np.angle(cropped_ey)) / np.exp(1j * np.angle(cropped_ex)))

        # Compute Correction Phase
        if modulation_type == 'Azimuthal Polarization':
            # Take the current phase as the maximum deviated phase
            intensity_threshold = 0.75 * (cropped_s0.max() - cropped_s0.min()) + cropped_s0.min()
            threshold_mask = cropped_s0 > intensity_threshold  # Locate light over noise
            ex_masked = cropped_ex * threshold_mask
            ey_masked = cropped_ey * threshold_mask
            ellipticity_map = compute_ellipticity([ex_masked, ey_masked], 'Jones')
            # Locate the diagonal parts of the beam, where we have both x and y electric field components
            both_contributions_mask = np.abs(cropped_ex) * np.abs(cropped_ey) / np.max( np.abs(cropped_ex) * np.abs(cropped_ey)) > 0.5
            # Compute the mean of the relative phase (mapped into the first quadrant) in the diagonal parts of the beam
            relative_phase_first_quadrant = map_angles_to_first_quadrant(relative_phase_cropped)
            current_phase = np.mean(relative_phase_first_quadrant[both_contributions_mask])

        else:
            # Compute the average phase of the illuminated part of the beam that surpasses an intensity threshold
            intensity_threshold = 0.2 * (cropped_s0.max() - cropped_s0.min()) + cropped_s0.min()
            current_phase = np.mean(relative_phase_cropped[cropped_s0 > intensity_threshold])

        correction_phase_map[ii, jj] = target_relative_phase - current_phase

    # Interpolation for fine resolution
    freqsX_mesh, freqsY_mesh = 1 / np.where(periodsX_mesh == 0, np.inf, periodsX_mesh), 1 / np.where(periodsY_mesh == 0,
                                                                                                     np.inf,
                                                                                                     periodsY_mesh)
    rbf_model = RBFInterpolator(np.column_stack((freqsX_mesh.ravel(), freqsY_mesh.ravel())),
                                correction_phase_map.ravel())

    fine_grid = [np.linspace(m.min(), m.max(), m.shape[0] * 4) for m in (freqsX_mesh, freqsY_mesh)]
    fine_freqsX_mesh, fine_freqsY_mesh = np.meshgrid(*fine_grid)

    corrective_phase_interpolation = rbf_model(
        np.column_stack((fine_freqsX_mesh.ravel(), fine_freqsY_mesh.ravel()))).reshape(fine_freqsX_mesh.shape)

    # Display correction maps
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    cbar_format = {'shrink': 0.6, 'orientation': 'vertical'}

    im1 = axes[0].imshow(correction_phase_map, vmin=-np.pi, vmax=np.pi, cmap='twilight')
    axes[0].set_title('Experimental Phase Map (rad)')
    fig.colorbar(im1, ax=axes[0], **cbar_format)

    im2 = axes[1].imshow(np.degrees(correction_phase_map), cmap='viridis')
    axes[1].set_title('Experimental Phase Map (°)(Norm. Cmap)')
    fig.colorbar(im2, ax=axes[1], **cbar_format)

    im3 = axes[2].imshow(np.degrees(corrective_phase_interpolation), cmap='viridis')
    axes[2].set_title('Interpolated Phase Map(°)(Norm. Cmap)')
    fig.colorbar(im3, ax=axes[2], **cbar_format)

    plt.show()

    # Generate and display Stokes collages
    if stokes_display == 'Maximum Projection':
        s0_max_proj = np.take_along_axis(s0_all, np.abs(s0_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)
        s1_max_proj = np.take_along_axis(s1_all, np.abs(s1_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)
        s2_max_proj = np.take_along_axis(s2_all, np.abs(s2_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)
        s3_max_proj = np.take_along_axis(s3_all, np.abs(s3_all).argmax(axis=0, keepdims=True), axis=0).squeeze(0)

        fig, _ = show_stokes_map(s0_max_proj, s1_max_proj/s0_max_proj, s2_max_proj/s0_max_proj, s3_max_proj/s0_max_proj,
                                 normalize=False)
        fig.suptitle('Maximum Projection')

        # Display Beams with its corresponding polarization ellipses
        polarization_map_analysis([s0_max_proj, s1_max_proj, s2_max_proj, s3_max_proj], vector_tpye='Stokes',
                                  show_figure=True, show_relative_phase_map=True,
                                  draw_step=1, filter_arrow_intensity=True)

    elif stokes_display == 'Collage':
        stokes_collages = [make_collage(param, n_steps) for param in [cropped_s0_list, cropped_s1_norm_list, cropped_s2_norm_list, cropped_s3_norm_list]]
        fig, _ = show_stokes_map(*stokes_collages, normalize=False)
        fig.suptitle('Collage, positions not to scale')

        # Display Beams with its corresponding polarization ellipses
        polarization_map_analysis([*stokes_collages], vector_tpye='Stokes',
                                  show_figure=True, show_relative_phase_map=True,
                                  draw_step=1, filter_arrow_intensity=True)

    # Compute and display max projection
    projection = np.max(np.array(s0_all, dtype=np.float64), axis=0)
    plt.figure()
    plt.title(f'S0 Maximum Projection: {modulation_type}, Subpixel Size: {eff_pixel_size}')
    plt.imshow(projection, cmap='gray', extent=(0, projection.shape[1] * camera_pixel_size_mm,
                                                projection.shape[0] * camera_pixel_size_mm, 0))
    plt.xlabel('mm')
    plt.ylabel('mm')
    plt.show()

    # Display 3D viewer of measurements
    Imshow3D(np.array([measure[ii] for measure in measurements for ii in range(len(measure))], dtype=np.float64))

    if return_max_projection_stokes:
        return correction_phase_map, rbf_model, s0_max_proj, s1_max_proj, s2_max_proj, s3_max_proj
    else:
        return correction_phase_map, rbf_model


if __name__ == "__main__":
    dataset_filepath = ask_files_location('Select Pickle File for Dataset', return_first_string = True)
    # noinspection PyTypeChecker
    with open(dataset_filepath, 'rb') as file:
        dictionary = pickle.load(file)
        # Unpack values from dictionary
        list_single_experiment_measures_images = dictionary['List of All Measures']
        camera_pixel_size_mm_experiment = dictionary['Camera Pixel Size (mm)']
        periodsX_mesh_experiment = dictionary['Periods X Meshgrid']
        periodsY_mesh_experiment = dictionary['Periods Y Meshgrid']
        n_steps_experiment = dictionary['Experiment Configuration']['N Scan Steps']
        characterize_polarization_experiment = dictionary['Experiment Configuration']['Characterize Polarization']
        modulation_type_experiment = dictionary['Experiment Configuration']['Modulation Type']
        eff_pixel_size_experiment = dictionary['Experiment Configuration']['Effective Arrizon Pixel Size (Pix)']
        target_fov_microscope_m = dictionary['Experiment Configuration']['Target FOV Microscope (m)']
        roi_size_experiment = 20

        correction_phase_map_experiment, rbf_model_exp = compute_polarization_correction_map(list_single_experiment_measures_images, modulation_type_experiment,
                                                                                             n_steps_experiment,
                                                                                             eff_pixel_size_experiment,
                                                                                             camera_pixel_size_mm_experiment,
                                                                                             roi_size_experiment, periodsX_mesh_experiment, periodsY_mesh_experiment,
                                                                                             'Maximum Projection')
        plt.pause(0.01)

        # Save if desired
        save_data = ask_yes_no_prompt('Save Calibration Object?', default_answer='no')
        if save_data:
            sanitized_modulation_type_str = modulation_type_experiment.replace(" ", "_").lower()
            saving_filename = f'Correction_Object_{sanitized_modulation_type_str}_FOV{int(target_fov_microscope_m/um)}um_{n_steps_experiment}x{n_steps_experiment}.pkl'
            saving_filename = add_date_prefix(saving_filename)
            saving_filename = get_unique_filename(saving_filename)
            with open(f'{saving_filename}', 'wb') as saving_file:
                total_dictionary = {
                    'Experimental Corrective Phase Map': correction_phase_map_experiment,
                    'Experiment Dictionary': dictionary,
                    'Interpolation Model Correction Phase Map': rbf_model_exp
                }
                # noinspection PyTypeChecker
                pickle.dump(total_dictionary, saving_file)
            print(f'File successfully saved to {saving_filename}')
