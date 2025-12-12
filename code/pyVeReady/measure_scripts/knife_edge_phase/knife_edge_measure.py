from pyVeReady.holo_generation.hologram_computation_V2 import *
from pyVeReady.holo_generation.holo_generation_functions import *
import slmpy
from tqdm import tqdm
from scipy.special import erf
from scipy.optimize import curve_fit
from pyVeReady.hardware_control.micromanager_camera_python import *
from pyVeReady.utils.image_utils import *
import scienceplots

if __name__ == '__main__':
    #%% EXPERIMENTAL ACQUISITION
    # Measure Parameters
    pixel_steps = 10
    show_live_acquisition = False

    # Hologram Computation Parameters
    double_pass = True
    rowsSLM, colsSLM = 1024, 1280
    laser_wavelength_nm = 488
    lin_period_X = -16
    lin_period_Y = 0
    pupil_diameter_mm = 10
    obl_ast_coeff = 0
    vert_ast_coeff = 0
    coeff = 0
    use_A_display, use_B_display = True, False
    slm_correction_type, slm_correction_filepath = 'No Correction', None

    # Initialize SLM and display only a linear phase hologram
    slm = slmpy.SLMdisplay()
    holo, _, _ = compute_hologram('Zeros Hologram', double_pass, rowsSLM, colsSLM, laser_wavelength_nm,
                                  lin_period_X, lin_period_Y, pupil_diameter_mm, obl_ast_coeff, vert_ast_coeff,
                                  0, 0, 0, 0, True, False,
                                  slm_correction_type, slm_correction_filepath, coeff, 0, verbose = False)
    slm.updateArray(holo)
    time.sleep(1)

    # Initialize Camera and set exposure time with full beam
    camera = PyMicroManagerPlusCamera('TIS Camera')
    camera.connect()
    camera.set_exposure_interactive()

    # Measure in the following manner:
    # First we only use the A display and iterate through the offsets in X (cols) and Y (rows) in that order
    # Secondly we only use the B display and iterate through the offsets in X (cols) and Y (rows) in that order

    slm_canvas = np.zeros((rowsSLM, colsSLM))
    A_slm, B_slm, _ = split_array(slm_canvas)
    rows_semi_display, cols_semi_display = A_slm.shape
    rows_offsets_values = np.arange(0, rows_semi_display, pixel_steps) - rows_semi_display // 2
    cols_offsets_values = np.arange(0, cols_semi_display, pixel_steps) - cols_semi_display // 2

    if show_live_acquisition:
        # Create figures for SLM hologram and captured image
        fig_slm, ax_slm = plt.subplots()
        im_obj_slm = ax_slm.imshow(np.zeros((rowsSLM, colsSLM)), cmap = 'gray', vmin = 0, vmax = 255)
        ax_slm.set_title("SLM Hologram")
        plt.show(block = False)

        fig_cam, ax_cam = plt.subplots()
        im_obj_cam = ax_cam.imshow(np.zeros((480, 744)), cmap = 'gray', vmin = 0, vmax = 255)
        ax_cam.set_title("Captured Image")
        plt.show(block = False)

    # Define scan cases
    scan_cases = [
        ("A Display X Scan", True, False, 'Knife Edge X (Phase)', cols_offsets_values, True),
        ("A Display Y Scan", True, False, 'Knife Edge Y (Phase)', rows_offsets_values, False),
        ("B Display X Scan", False, True, 'Knife Edge X (Phase)', cols_offsets_values, True),
        ("B Display Y Scan", False, True, 'Knife Edge Y (Phase)', rows_offsets_values, False)
    ]

    images_list = []
    for scan_label, use_A_display, use_B_display, modulation_type, offsets, is_col in scan_cases:
        for offset in tqdm(offsets, desc = scan_label):
            xoffA = offset if is_col and use_A_display else 0
            yoffA = offset if not is_col and use_A_display else 0
            xoffB = offset if is_col and use_B_display else 0
            yoffB = offset if not is_col and use_B_display else 0

            holo, _, _ = compute_hologram(modulation_type, double_pass, rowsSLM, colsSLM, laser_wavelength_nm,
                                          lin_period_X, lin_period_Y, pupil_diameter_mm, obl_ast_coeff, vert_ast_coeff,
                                          xoffA, yoffA, xoffB, yoffB, use_A_display, use_B_display,
                                          slm_correction_type, slm_correction_filepath, coeff, 0, verbose = False)

            slm.updateArray(holo)  # Send hologram to SLM

            if show_live_acquisition:
                # Update SLM figure
                im_obj_slm.set_data(holo)
                fig_slm.canvas.draw()
                fig_slm.canvas.flush_events()

            time.sleep(0.85)  # Maintain consistent timing for SLM updating

            # Capture image from camera
            im = camera.snap()
            images_list.append(im.astype(np.float64))

            if show_live_acquisition:
                # Update Captured Image figure
                im_obj_cam.set_data(im)
                fig_cam.canvas.draw()
                fig_cam.canvas.flush_events()

    slm.close()
    camera.disconnect()

    #%% DATA PROCESSING

    # Compute Total Intensities and Split Data for each display
    total_intensities = [np.sum(image) for image in images_list]
    total_intensities = np.array(total_intensities, dtype = np.float64)
    total_intensities_A, total_intensities_B = np.split(total_intensities, 2)
    total_intensities_A = total_intensities_A / total_intensities_A.max()
    total_intensities_B = total_intensities_B / total_intensities_B.max()

    total_intensities_A_scan_X = total_intensities_A[:len(cols_offsets_values)]
    total_intensities_A_scan_Y = total_intensities_A[len(cols_offsets_values):]
    total_intensities_B_scan_X = total_intensities_B[:len(cols_offsets_values)]
    total_intensities_B_scan_Y = total_intensities_B[len(cols_offsets_values):]

    # Perform fitting
    def knife_edge_intensity(offset, A, w, center, d):
        return d + A / 2 * erf((offset - center) / (w / np.sqrt(2)))

    initial_guess = (1, 100, 0, 0)
    fits = {label: curve_fit(knife_edge_intensity, offsets, intensities, initial_guess)[0]
            for label, offsets, intensities in zip(
                                                ['x_A', 'y_A', 'x_B', 'y_B'],
                                                [cols_offsets_values, rows_offsets_values, cols_offsets_values, rows_offsets_values],
                                                [total_intensities_A_scan_X, total_intensities_A_scan_Y,
                                                 total_intensities_B_scan_X, total_intensities_B_scan_Y]
        )}

    # Plot results
    plt.style.use(['science','no-latex'])
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharex = True, sharey = True)
    for i, (display, x_fit, y_fit, x_int, y_int) in enumerate([
            ("A Display", fits['x_A'], fits['y_A'], total_intensities_A_scan_X, total_intensities_A_scan_Y),
            ("B Display", fits['x_B'], fits['y_B'], total_intensities_B_scan_X, total_intensities_B_scan_Y)]):

        ax = axes[i]
        ax.plot(cols_offsets_values, x_int, 'xb', label='X')
        ax.plot(rows_offsets_values, y_int, 'xr', label='Y')
        ax.plot(cols_offsets_values, knife_edge_intensity(cols_offsets_values, *x_fit), 'b')
        ax.plot(rows_offsets_values, knife_edge_intensity(rows_offsets_values, *y_fit), 'r')
        ax.scatter(x_fit[2], knife_edge_intensity(x_fit[2], *x_fit))
        ax.scatter(y_fit[2], knife_edge_intensity(y_fit[2], *y_fit))
        ax.set(title = display, xlabel = 'Offsets', ylabel = 'Intensity (a.u.)')
        ax.legend()

    plt.tight_layout()
    plt.show()

    viewer = Imshow3D(np.array(images_list, dtype = np.float64))

    # Print results
    for key in ['x_A', 'y_A', 'x_B', 'y_B']:
        print(f'{key.replace("_", " ").upper()} Center: {fits[key][2]:.4f}, Width: {fits[key][1]:.4f}')