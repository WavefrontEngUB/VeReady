from pyVeReady.hardware_control.micromanager_camera_python import *
from pyVeReady.hardware_control.ell14_motor_control import *
from pyVeReady.hardware_control.IO_Matchbox_Laser_Control import *
from pyVeReady.hardware_control.misc import measure_polarization_images
from pyVeReady.holo_generation.hologram_computation_V2 import *
from pyVeReady.measure_scripts.beam_focal_scan_linear_phase.analyze_datasets import *
import numpy as np
import time
import slmpy
from tqdm.auto import tqdm
import pickle
import matplotlib
matplotlib.use('Tkagg')

# Constants
mm, um, nm = 1e-3, 1e-6, 1e-9

# Hardware Parameters
camera_exposure_time_ms = 5
power_LG = 0.65
power_circ_right = 0.28
power_azimuthal = 1.6

x_off_A, y_off_A = 7, -52
x_off_B, y_off_B = -8, 51
slm_shape, slm_pixel_size = (1024, 1280), 12.5 * um
slm_correction_type = 'Double-Pass Correction'
slm_correction_filepath = r'C:\Users\Usuario\OneDrive - Universitat de Barcelona\PhD\VeReady\Code\Vector_Beams_LabVIEW\CAL_LSH0804783_490nm.bmp'

pol_correction_filepath = r'C:\Users\Usuario\OneDrive - Universitat de Barcelona\PhD\VeReady\Code\Vector_Beams_LabVIEW\PySLM_VeReady\pyVeReady\measure_scripts\beam_focal_scan_linear_phase\20250327_Correction_Object_circular_right_FOV4um_5x5.pkl'

characterize_pol = True
default_scan_steps = 5
default_effective_arrizon_pixel_size = 3

config_circular_right_no_pol_correction = {
    'Experiment Name': 'circ_right_no_pol_correction',
    "Modulation Type": 'Circular Right', 'Use Pol. Correction': False, 'Pol. Correction Filepath': pol_correction_filepath,
    'Central Period X Pix': -20.5, 'Central Period Y Pix': np.inf,
    'Characterize Polarization': characterize_pol, 'N Angles Rotating QWP': 8,
    'Target FOV Microscope (m)': 2 * 2 * um, 'N Scan Steps': default_scan_steps,
    'Objective Lens Focal (m)': 2 * mm, 'Tube Lens Focal (m)': 200 * mm, 'Scan Lens Focal (m)': 150 * mm,
    'SLM Correction Type': slm_correction_type, 'SLM Correction Filepath': slm_correction_filepath,
    'Pupil Diameter (mm)': 5.75, "Effective Arrizon Pixel Size (Pix)": default_effective_arrizon_pixel_size,
    'SLM Shape': slm_shape, 'SLM Pixel Size (m)': slm_pixel_size,
    'X Offset A': x_off_A, 'Y Offset A': y_off_A,
    'X Offset B': x_off_B, 'Y Offset B': y_off_B,
    'Laser Wavelength (m)': 488 * nm, "Laser Power (mW)": power_circ_right
    }

config_circ_right_with_pol_correction = {
    'Experiment Name': 'circ_right_with_pol_correction',
    "Modulation Type": 'Circular Right', 'Use Pol. Correction': True,
    'Pol. Correction Filepath': pol_correction_filepath,
    'Central Period X Pix': -20.5, 'Central Period Y Pix': np.inf,
    'Characterize Polarization': characterize_pol, 'N Angles Rotating QWP': 8,
    'Target FOV Microscope (m)': 2 * 2 * um, 'N Scan Steps': default_scan_steps,
    'Objective Lens Focal (m)': 2 * mm, 'Tube Lens Focal (m)': 200 * mm, 'Scan Lens Focal (m)': 150 * mm,
    'SLM Correction Type': slm_correction_type, 'SLM Correction Filepath': slm_correction_filepath,
    'Pupil Diameter (mm)': 5.75, "Effective Arrizon Pixel Size (Pix)": default_effective_arrizon_pixel_size,
    'SLM Shape': slm_shape, 'SLM Pixel Size (m)': slm_pixel_size,
    'X Offset A': x_off_A, 'Y Offset A': y_off_A,
    'X Offset B': x_off_B, 'Y Offset B': y_off_B,
    'Laser Wavelength (m)': 488 * nm, "Laser Power (mW)": power_circ_right
    }

config_LG_right_no_pol_correction = {
    'Experiment Name': 'LG_right_no_pol_correction',
    "Modulation Type": 'Laguerre-Gaussian Right', 'Use Pol. Correction': False, 'Pol. Correction Filepath': pol_correction_filepath,
    'Central Period X Pix': -20.5, 'Central Period Y Pix': np.inf,
    'Characterize Polarization': characterize_pol, 'N Angles Rotating QWP': 8,
    'Target FOV Microscope (m)': 2 * 2 * um, 'N Scan Steps': default_scan_steps,
    'Objective Lens Focal (m)': 2 * mm, 'Tube Lens Focal (m)': 200 * mm, 'Scan Lens Focal (m)': 150 * mm,
    'SLM Correction Type': slm_correction_type, 'SLM Correction Filepath': slm_correction_filepath,
    'Pupil Diameter (mm)': 5.75, "Effective Arrizon Pixel Size (Pix)": default_effective_arrizon_pixel_size,
    'SLM Shape': slm_shape, 'SLM Pixel Size (m)': slm_pixel_size,
    'X Offset A': x_off_A, 'Y Offset A': y_off_A,
    'X Offset B': x_off_B, 'Y Offset B': y_off_B,
    'Laser Wavelength (m)': 488 * nm, "Laser Power (mW)": power_LG
    }

config_LG_right_with_pol_correction = {
    'Experiment Name': 'LG_right_with_pol_correction',
    "Modulation Type": 'Laguerre-Gaussian Right', 'Use Pol. Correction': True,
    'Pol. Correction Filepath': pol_correction_filepath,
    'Central Period X Pix': -20.5, 'Central Period Y Pix': np.inf,
    'Characterize Polarization': characterize_pol, 'N Angles Rotating QWP': 8,
    'Target FOV Microscope (m)': 2 * 2 * um, 'N Scan Steps': default_scan_steps,
    'Objective Lens Focal (m)': 2 * mm, 'Tube Lens Focal (m)': 200 * mm, 'Scan Lens Focal (m)': 150 * mm,
    'SLM Correction Type': slm_correction_type, 'SLM Correction Filepath': slm_correction_filepath,
    'Pupil Diameter (mm)': 5.75, "Effective Arrizon Pixel Size (Pix)": default_effective_arrizon_pixel_size,
    'SLM Shape': slm_shape, 'SLM Pixel Size (m)': slm_pixel_size,
    'X Offset A': x_off_A, 'Y Offset A': y_off_A,
    'X Offset B': x_off_B, 'Y Offset B': y_off_B,
    'Laser Wavelength (m)': 488 * nm, "Laser Power (mW)": power_LG
    }
config_azimuthal_no_pol_correction = {
    'Experiment Name': 'azimuthal_no_pol_correction',
    "Modulation Type": 'Azimuthal Polarization', 'Use Pol. Correction': False,
    'Pol. Correction Filepath': pol_correction_filepath,
    'Central Period X Pix': -20.5, 'Central Period Y Pix': np.inf,
    'Characterize Polarization': characterize_pol, 'N Angles Rotating QWP': 8,
    'Target FOV Microscope (m)': 2 * 2 * um, 'N Scan Steps': default_scan_steps,
    'Objective Lens Focal (m)': 2 * mm, 'Tube Lens Focal (m)': 200 * mm, 'Scan Lens Focal (m)': 150 * mm,
    'SLM Correction Type': slm_correction_type, 'SLM Correction Filepath': slm_correction_filepath,
    'Pupil Diameter (mm)': 5.75, "Effective Arrizon Pixel Size (Pix)": default_effective_arrizon_pixel_size,
    'SLM Shape': slm_shape, 'SLM Pixel Size (m)': slm_pixel_size,
    'X Offset A': x_off_A, 'Y Offset A': y_off_A,
    'X Offset B': x_off_B, 'Y Offset B': y_off_B,
    'Laser Wavelength (m)': 488 * nm, "Laser Power (mW)": power_azimuthal
    }
config_azimuthal_with_pol_correction = {
    'Experiment Name': 'azimuthal_with_pol_correction',
    "Modulation Type": 'Azimuthal Polarization', 'Use Pol. Correction': True,
    'Pol. Correction Filepath': pol_correction_filepath,
    'Central Period X Pix': -20.5, 'Central Period Y Pix': np.inf,
    'Characterize Polarization': characterize_pol, 'N Angles Rotating QWP': 8,
    'Target FOV Microscope (m)': 2 * 2 * um, 'N Scan Steps': default_scan_steps,
    'Objective Lens Focal (m)': 2 * mm, 'Tube Lens Focal (m)': 200 * mm, 'Scan Lens Focal (m)': 150 * mm,
    'SLM Correction Type': slm_correction_type, 'SLM Correction Filepath': slm_correction_filepath,
    'Pupil Diameter (mm)': 5.75, "Effective Arrizon Pixel Size (Pix)": default_effective_arrizon_pixel_size,
    'SLM Shape': slm_shape, 'SLM Pixel Size (m)': slm_pixel_size,
    'X Offset A': x_off_A, 'Y Offset A': y_off_A,
    'X Offset B': x_off_B, 'Y Offset B': y_off_B,
    'Laser Wavelength (m)': 488 * nm, "Laser Power (mW)": power_azimuthal
    }
config_circ_right_calibration = {
    'Experiment Name': 'calibration_circ_right',
    "Modulation Type": 'Circular Right',           'Use Pol. Correction': False,               'Pol. Correction Filepath': None,
    'Central Period X Pix': -20.5,                 'Central Period Y Pix': np.inf,
    'Characterize Polarization': characterize_pol, 'N Angles Rotating QWP': 8,
    'Target FOV Microscope (m)': 2 * 2 * um,       'N Scan Steps': default_scan_steps,
    'Objective Lens Focal (m)': 2 * mm,            'Tube Lens Focal (m)': 200 * mm,            'Scan Lens Focal (m)': 150 * mm,
    'SLM Correction Type': slm_correction_type, 'SLM Correction Filepath': slm_correction_filepath,
    'Pupil Diameter (mm)': 5.75,                   "Effective Arrizon Pixel Size (Pix)": default_effective_arrizon_pixel_size,
    'SLM Shape': slm_shape, 'SLM Pixel Size (m)': slm_pixel_size,
    'X Offset A': x_off_A, 'Y Offset A': y_off_A,
    'X Offset B': x_off_B, 'Y Offset B': y_off_B,
    'Laser Wavelength (m)': 488 * nm,              "Laser Power (mW)": power_circ_right
    }

experiment_configs = [config_azimuthal_with_pol_correction
]

# Save Data
save_data = True

# Initialize Hardware
slm = slmpy.SLMdisplay()
laser = IntegratedOpticsMatchbox(port='COM4')
laser.start_laser()
camera = PyMicroManagerPlusCamera('TIS Camera')
camera.connect()
camera.set_exposure(camera_exposure_time_ms)
camera_pixel_size_mm = (camera.get_pixel_size_um() * um) / mm
if characterize_pol:
    devices = init_and_scan_devices('COM3')
    motor_qwp = AddressedDevice(devices[0])
else:
    motor_qwp = None

# Initialize progress bar
total_steps = sum(config['N Scan Steps']**2 for config in experiment_configs)  # Total number of steps
pbar = tqdm(total=total_steps, desc="Experiment Progress", unit="step")

# Data Collection
list_of_all_measures = []
for config in experiment_configs:
    # Load Config Parameters
    experiment_name = config['Experiment Name']
    modulation_type = config['Modulation Type']
    use_pol_correction = config['Use Pol. Correction']
    pol_correction_filepath = config['Pol. Correction Filepath']
    central_period_x_pix = config['Central Period X Pix']
    central_period_y_pix = config['Central Period Y Pix']
    characterize_polarization = config['Characterize Polarization']
    n_angles_rotating_qwp = config['N Angles Rotating QWP']
    target_fov_microscope_m = config['Target FOV Microscope (m)']
    n_scan_steps = config['N Scan Steps']
    objective_lens_focal_m = config['Objective Lens Focal (m)']
    tube_lens_focal_m = config['Tube Lens Focal (m)']
    scan_lens_focal_m = config['Scan Lens Focal (m)']
    slm_correction_type = config['SLM Correction Type']
    slm_correction_filepath = config['SLM Correction Filepath']
    pupil_diameter_mm = config['Pupil Diameter (mm)']
    effective_pixel_size = config['Effective Arrizon Pixel Size (Pix)']
    slm_shape = config['SLM Shape']
    slm_pixel_size_m = config['SLM Pixel Size (m)']
    x_offset_A = config['X Offset A']
    y_offset_A = config['Y Offset A']
    x_offset_B = config['X Offset B']
    y_offset_B = config['Y Offset B']
    laser_wavelength_m = config['Laser Wavelength (m)']
    laser_power_mw = config['Laser Power (mW)']

    # Adjust Hardware Settings
    motor_qwp.set_jog_step(180 / n_angles_rotating_qwp)
    laser.set_opt_power(laser_power_mw)
    time.sleep(0.25)

    # Compute Scanning Linear Phases
    max_relative_deflection = (target_fov_microscope_m / (2*objective_lens_focal_m)) * (tube_lens_focal_m / scan_lens_focal_m)
    central_deflection_X = period_to_deflection_angle(central_period_x_pix, laser_wavelength_m, slm_pixel_size_m)
    central_deflection_Y = period_to_deflection_angle(central_period_y_pix, laser_wavelength_m, slm_pixel_size_m)
    deflection_angles_X = np.linspace(central_deflection_X - max_relative_deflection, central_deflection_X + max_relative_deflection, n_scan_steps)
    deflection_angles_Y = np.linspace(central_deflection_Y - max_relative_deflection, central_deflection_Y + max_relative_deflection, n_scan_steps)
    periods_X = deflection_angle_to_period(deflection_angles_X, laser_wavelength_m, slm_pixel_size_m)
    periods_Y = deflection_angle_to_period(deflection_angles_Y, laser_wavelength_m, slm_pixel_size_m)
    periods_X_meshgrid, periods_Y_meshgrid = np.meshgrid(periods_X, periods_Y)

    # Initialize the figure for real-time updates
    fig, ax = plt.subplots()
    plt.ion()
    im_display = ax.imshow(np.zeros(slm_shape), cmap='gray', vmin=0, vmax=255)  # Initial empty image
    plt.title("")
    plt.show(block=False)

    # Scan the Beam
    list_single_experiment_measures = []
    for ii, jj in np.ndindex(periods_X_meshgrid.shape):
        lin_period_X, lin_period_Y = periods_X_meshgrid[ii, jj], periods_Y_meshgrid[ii, jj]
        # Compute hologram and update SLM
        holo, _, _ = compute_hologram(modulation_type, True, slm_shape[0], slm_shape[1], laser_wavelength_m / nm,
                                        lin_period_X, lin_period_Y, pupil_diameter_mm, 0, effective_pixel_size,
                                        x_offset_A, y_offset_A, x_offset_B, y_offset_B, True, True,
                                        slm_correction_type, slm_correction_filepath, 0, 0,
                                        use_pol_correction, pol_correction_filepath)
        slm.updateArray(holo)

        if characterize_polarization:
            # Allow time for the SLM update
            time.sleep(0.4)

            # Measure polarization images
            polarization_images_list = measure_polarization_images(motor_qwp, camera, n_angles_rotating_qwp)
            list_single_experiment_measures.append(polarization_images_list)

            # Update the figure
            im_display.set_data(np.mean(np.array(polarization_images_list, dtype=np.float64), axis=0))  # Update image data
            ax.set_title(f"Mean Image - X: {lin_period_X:.2f}, Y: {lin_period_Y:.2f}")  # Update title
            plt.draw()  # Redraw figure
            plt.pause(0.001)  # Small pause to allow GUI update
        else:
            # Allow time for the SLM update
            time.sleep(0.5)

            # Measure Intensity
            intensity = camera.snap()
            list_single_experiment_measures.append(intensity)

            # Update the figure
            im_display.set_data(intensity)  # Update image data
            ax.set_title(f"Intensity Image - X: {lin_period_X:.2f}, Y: {lin_period_Y:.2f}")  # Update title
            plt.draw()  # Redraw figure
            plt.pause(0.001)  # Small pause to allow GUI update

        # Update progress bar
        pbar.update(1)

    plt.close(fig)
    list_of_all_measures.append(list_single_experiment_measures)

    # Save Data
    saving_filename = f'{experiment_name}_FOV{int(target_fov_microscope_m / um)}um_{n_scan_steps}x{n_scan_steps}.pkl'
    saving_filename = add_date_prefix(saving_filename)
    saving_filename = get_unique_filename(saving_filename)
    with open(saving_filename, 'wb') as file:
        dictionary = {
            'List of All Measures': list_single_experiment_measures,
            'Experiment Configuration': config,
            'Scan Step (nm)': (target_fov_microscope_m / n_scan_steps) / nm,
            'Periods X Meshgrid': periods_X_meshgrid,
            'Periods Y Meshgrid': periods_Y_meshgrid,
            'Camera Pixel Size (mm)': camera_pixel_size_mm
        }
        # noinspection PyTypeChecker
        pickle.dump(dictionary, file)
        print(f'File successfully saved to {saving_filename}')
