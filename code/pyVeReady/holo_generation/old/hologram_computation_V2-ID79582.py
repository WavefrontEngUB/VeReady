# import sys
# import pickle
#
# # Add path so LabVIEW also recognizes pyVeReady as a module
# sys.path.append(r'C:\Users\laboratori\OneDrive - Universitat de Barcelona\PhD\VeReady\Code\Vector_Beams_LabVIEW\PySLM_VeReady')
# from pyVeReady.holo_generation.holo_generation_functions import *
# from pyVeReady.utils.image_utils import *
# # from scipy.ndimage import rotate
# np.random.seed(1)
#
# def dummy_function():
#     return
#
# def compute_hologram(modulation_type, double_pass, rowsSLM, colsSLM, laser_wavelength_nm,
#                      lin_period_X, lin_period_Y,
#                      pupil_diameter_mm, obl_ast_coeff, eff_pixel_size,
#                      xoffA, yoffA, xoffB, yoffB,
#                      use_A_display, use_B_display,
#                      slm_correction_type, slm_correction_filepath,
#                      coeff,
#                      defocus_mm,
#                      fine_tune_phase_deg,
#                      use_polarization_correction=False, polarization_correction_filepath=None,
#                      verbose = False):
#
#     # Initialize variables to avoid errors
#     holo_x, holo_y, c_holo_x, c_holo_y, A_slm, B_slm, concat_ax = None, None, None, None, None, None, None
#
#     # Locate root of the script
#     root = os.path.abspath(os.path.dirname(__file__))
#
#     # Define Constants
#     mm, um, nm = 1e-3, 1e-6, 1e-9
#     slm_specsheet_wavelength_nm = 490
#     slm_gray_level_2pi_at_specsheet_wavelength = 196
#     slm_pixel_size_um = 12.5
#     laser_wavelength = laser_wavelength_nm * nm
#     focusing_lens_focal = 300 * mm
#
#     # Compute SLM LUTs bounds
#     gray_level_0 = 0
#     gray_level_2pi = int(laser_wavelength_nm / slm_specsheet_wavelength_nm * slm_gray_level_2pi_at_specsheet_wavelength)
#
#     # Initialize Modulation Manager
#     slm_modulations = ModulationManager(
#         rowsSLM=rowsSLM,
#         colsSLM=colsSLM,
#         double_pass=double_pass,
#         gray_level_zero=gray_level_0,
#         gray_level_two_pi_span=gray_level_2pi,
#         eff_pixel_size=eff_pixel_size,
#         slm_pixel_size_um=slm_pixel_size_um,
#         laser_wavelength_nm=laser_wavelength_nm,
#         pupil_diameter_mm=pupil_diameter_mm
#     )
#     slm_modulations.generate_beam_shaping_hologram(modulation_type)
#     slm_modulations.add_linear_phase(lin_period_X, lin_period_Y)
#     slm_modulations.add_spherical_wavefront(focusing_lens_focal, defocus_mm)
#     slm_modulations.add_astigmatism_phase(obl_ast_coeff)
#     slm_modulations.add_pupil_mask()
#
#     c_holo_x = slm_modulations.c_holo_x
#     c_holo_y = slm_modulations.c_holo_y
#
#     # Polarization Correction
#     if use_polarization_correction:
#         if polarization_correction_filepath is None:
#             raise ValueError("polarization_correction_filepath cannot be None")
#         else:
#             with open(polarization_correction_filepath, 'rb') as file:
#                 dictionary = pickle.load(file)
#                 interp_model = dictionary['Interpolation Model Correction Phase Map']
#                 corrective_phase = interp_model(np.array([[1/lin_period_X, 1/lin_period_Y]]))
#
#                 c_holo_y = c_holo_y * np.exp(1j * corrective_phase)
#
#     # Fine Tune Relative phase
#     c_holo_y *= np.exp(1j * fine_tune_phase_deg * np.pi/180)
#
#     # Process Holograms for Display
#     if double_pass:
#         c_holo_x_displayed = slice_offset_part(+c_holo_x, slm_modulations.B_slm.shape[0], slm_modulations.B_slm.shape[1], xoffB, yoffB) # B display modulates E_x in our setup
#         c_holo_y_displayed = slice_offset_part(+c_holo_y, slm_modulations.A_slm.shape[0], slm_modulations.A_slm.shape[1], xoffA, yoffA) # A display modulates E_y in our setup
#
#         holo_x_displayed = c_phase2gray_interp(c_holo_x_displayed, slm_modulations.slm_lut.gray_level_lut, slm_modulations.slm_lut.phase_lut)
#         holo_y_displayed = c_phase2gray_interp(c_holo_y_displayed, slm_modulations.slm_lut.gray_level_lut, slm_modulations.slm_lut.phase_lut)
#
#         if not use_A_display:
#             c_holo_y_displayed = np.zeros_like(c_holo_y_displayed)
#             holo_y_displayed = np.zeros_like(holo_y_displayed, dtype=np.uint8)
#         if not use_B_display:
#             c_holo_x_displayed = np.zeros_like(c_holo_x_displayed)
#             holo_x_displayed = np.zeros_like(holo_x_displayed, dtype=np.uint8)
#
#         c_holo_SLM = np.concatenate((c_holo_y_displayed,
#                                     np.flip(np.flip(c_holo_x_displayed, axis=0), axis=1)), # Flip due to telescope in our setup
#                                     axis = slm_modulations.concat_ax) #  Concatenate A display (E_y) with B display (E_x)
#         holo_SLM = np.concatenate((holo_y_displayed,
#                                   np.flip(np.flip(holo_x_displayed, axis=0), axis=1)), # Flip due to telescope in our setup
#                                   axis = slm_modulations.concat_ax) #  Concatenate A display (E_y) with B display (E_x)
#     else:
#         c_holo_x_displayed = slice_offset_part(c_holo_x, rowsSLM, colsSLM, xoffA, yoffA)
#         holo_x_displayed = c_phase2gray_interp(c_holo_x_displayed, slm_modulations.slm_lut.gray_level_lut, slm_modulations.slm_lut.phase_lut)
#
#         if not use_A_display:
#             holo_x_displayed = np.zeros_like(holo_x_displayed, dtype=np.uint8)
#
#         c_holo_SLM = c_holo_x_displayed
#         holo_SLM = holo_x_displayed
#
#     # Apply SLM corrections
#     if slm_correction_type == 'Direct Correction':
#         correction_map = plt.imread(slm_correction_filepath)
#         correction_map_phase = pad2size(correction_map, rowsSLM, colsSLM) / correction_map.max() * 2 * np.pi
#         c_holo_SLM = c_holo_SLM * np.exp(1j * correction_map_phase)
#         c_holo_SLM[np.abs(c_holo_SLM) == 0] = 0  # To avoid -0, +0 phase ambiguity
#         holo_SLM = c_phase2gray_interp(c_holo_SLM, slm_modulations.slm_lut.gray_level_lut, slm_modulations.slm_lut.phase_lut)
#
#     if slm_correction_type == 'Double-Pass Correction':
#         correction_map_phase = generate_double_pass_correction_map(slm_correction_filepath, rowsSLM, colsSLM,
#                                                                    gray_level_2pi, int(1000 * 0.5 * (pupil_diameter_mm / slm_pixel_size_um)),
#                                                                    xoffA, yoffA, xoffB, yoffB)
#
#         c_holo_SLM = c_holo_SLM * np.exp(1j * correction_map_phase)
#         c_holo_SLM[np.abs(c_holo_SLM) == 0] = 0  # To avoid -0, +0 phase ambiguity
#         holo_SLM = c_phase2gray_interp(c_holo_SLM, slm_modulations.slm_lut.gray_level_lut, slm_modulations.slm_lut.phase_lut)
#
#     if modulation_type == 'Uniform Gray':
#         holo_SLM = np.uint8(np.ones([rowsSLM, colsSLM]) * coeff)
#
#     holo_SLM = np.uint8(holo_SLM)
#
#     if verbose:
#         return [holo_SLM, slm_modulations.slm_lut]
#     else:
#         return holo_SLM, np.float64(slm_modulations.slm_lut.phase_lut/np.pi), np.float64(slm_modulations.slm_lut.gray_level_lut)
#
# if __name__ == '__main__':
#     # Test the Function
#     modulation_type = 'Azimuthal Polarization'
#     double_pass = True
#     rowsSLM, colsSLM = 1024, 1280
#     laser_wavelength_nm = 488
#
#     lin_period_X, lin_period_Y = -20.5, 0
#     pupil_diameter_mm = 5.6
#     obl_ast_coeff, vert_ast_coeff = 0, 0
#     xoffA, yoffA, xoffB, yoffB = 0, 0, 0, 0
#     use_A_display, use_B_display = True, True
#     slm_correction_type, slm_correction_filepath = 'No Correction', None
#     coeff = -100
#
#
#     [holo_SLM, look_up_table] = compute_hologram(modulation_type, double_pass, rowsSLM, colsSLM, laser_wavelength_nm,
#                                                 lin_period_X, lin_period_Y,
#                                                 pupil_diameter_mm, obl_ast_coeff, 2,
#                                                 xoffA, yoffA, xoffB, yoffB,
#                                                 use_A_display, use_B_display,
#                                                 slm_correction_type, slm_correction_filepath,
#                                                 coeff,
#                                                 0,
#                                                 use_polarization_correction=False, polarization_correction_filepath=None,
#                                                 verbose=True)
#
#
#     plt.figure()
#     plt.imshow(holo_SLM, cmap = 'gray', vmin = 0, vmax = 255)
