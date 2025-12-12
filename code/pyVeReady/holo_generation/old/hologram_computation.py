# import sys
# # Add path so LabVIEW recog
# sys.path.append(r'C:\Users\Usuario\OneDrive - Universitat de Barcelona\PhD\VeReady\Code\Vector_Beams_LabVIEW\PySLM_VeReady')
#
# from holo_generation_functions import *
# from pyVeReady.utils.image_utils import *
# from scipy.ndimage import rotate
# np.random.seed(1)
#
# def dummy_function():
#     return
#
# def compute_hologram(modulation_type, double_pass, rowsSLM, colsSLM, laser_wavelength_nm,
#                      lin_period_X, lin_period_Y,F
#                      pupil_diameter_mm, obl_ast_coeff, vert_ast_coeff,
#                      xoffA, yoffA, xoffB, yoffB,
#                      use_A_display, use_B_display,
#                      slm_correction_type, slm_correction_filepath,
#                      coeff,
#                      verbose=False):
#
#     holo_x, holo_y, c_holo_x, c_holo_y, A_slm, B_slm = None, None, None, None, None, None
#     # Locate root of the script
#     root = os.path.abspath(os.path.dirname(__file__))
#
#     # Define Constants
#     mm, um = 1e-3, 1e-6
#     slm_specsheet_wavelength_nm = 490
#     slm_gray_level_2pi_at_specsheet_wavelength = 196
#     slm_pixel_size = 12.5 * um
#
#     # Compute SLM LUTs
#     gray_level_0 = 0
#     gray_level_2pi = int(slm_gray_level_2pi_at_specsheet_wavelength * laser_wavelength_nm / slm_specsheet_wavelength_nm)
#     (gray_level_lut, phase_lut,
#      m1_gray_lut, m2_gray_lut,
#      m1_phase_lut, m2_phase_lut,
#      full_complex_arrizon_lut) = compute_linear_slm_lut(gray_level_0, gray_level_2pi)
#
#     plt.figure()
#     plt.plot(gray_level_lut, phase_lut, 'b')
#
#     # Setup Displays dimensions and Coordinates definition
#     slm_blank = np.zeros([rowsSLM, colsSLM])  # Full SLM
#     if double_pass:
#         A_slm, B_slm, concat_ax = split_array(slm_blank)
#         X, Y, R, phi = pixel_coordinates(np.repeat(np.repeat(A_slm, 2, axis = 0), 2, axis = 1))
#     else:
#         X, Y, R, phi = pixel_coordinates(np.repeat(np.repeat(slm_blank, 2, axis = 0), 2, axis = 1))
#
#     X = rotate(X, 14, reshape = False, mode = 'nearest')
#     Y = rotate(Y, 14, reshape = False, mode = 'nearest')
#     R = rotate(R, 14, reshape = False, mode = 'nearest')
#     phi = rotate(phi, 14, reshape = False, mode = 'nearest')
#
#     X_arr, Y_arr = X[::2, ::2], Y[::2, ::2]  # Arrizon Coordinates
#     R_arr, phi_arr = np.sqrt(X_arr ** 2 + Y_arr ** 2), np.arctan2(Y_arr, X_arr)
#
#     # Compute target modulation
#     slm_modulations = Modulations(X, Y, R, phi, X_arr, Y_arr, R_arr, phi_arr, coeff, vert_ast_coeff, obl_ast_coeff)
#     ex_mod, ey_mod, encoding = slm_modulations.get_modulation(modulation_type)
#
#     # Compute holograms
#     ex_mod, ey_mod = np.complex128(ex_mod), np.complex128(ey_mod)
#     if encoding == 'arrizon':
#         holo_x, c_holo_x = compute_arrizon_holograms(ex_mod, m1_phase_lut, m2_phase_lut, m1_gray_lut, m2_gray_lut,
#                                                      full_complex_arrizon_lut)
#         holo_y, c_holo_y = compute_arrizon_holograms(ey_mod, m1_phase_lut, m2_phase_lut, m1_gray_lut, m2_gray_lut,
#                                                      full_complex_arrizon_lut)
#
#     elif encoding == 'phase':
#         c_holo_x = np.copy(ex_mod)
#         c_holo_y = np.copy(ey_mod)
#
#     # Add Phase Terms
#     pupil_radius_pixel = (pupil_diameter_mm / 2) / (slm_pixel_size / mm)
#     pupil_mask = R ** 2 <= pupil_radius_pixel ** 2
#
#     c_linphase_x = np.exp(1j * (2 * np.pi / lin_period_X) * X) if lin_period_X != 0 else 1
#     c_linphase_y = np.exp(1j * (2 * np.pi / lin_period_Y) * X) if lin_period_Y != 0 else 1
#     c_linear_phase = c_linphase_x * c_linphase_y
#
#     R_norm = np.sqrt((X / pupil_radius_pixel) ** 2 + (Y / pupil_radius_pixel) ** 2)
#     c_seidel_astigmatism = np.exp(
#         1j * (2 * np.pi / laser_wavelength_nm) * (obl_ast_coeff * laser_wavelength_nm) * R_norm ** 2 * np.cos(phi) ** 2) if obl_ast_coeff != 0 else 1
#
#     c_holo_x = c_holo_x * c_linear_phase * c_seidel_astigmatism * pupil_mask
#     c_holo_y = c_holo_y * c_linear_phase * c_seidel_astigmatism * pupil_mask
#
#     c_holo_x[np.abs(c_holo_x) == 0] = 0  # To avoid -0, +0 phase ambiguity
#     c_holo_y[np.abs(c_holo_y) == 0] = 0
#
#     # Process Holograms for Display
#     c_holo_x_displayed = slice_offset_part(-c_holo_x, B_slm.shape[0], B_slm.shape[1], xoffB, yoffB) # B display modulates E_x in our setup
#     c_holo_y_displayed = slice_offset_part(+c_holo_y, A_slm.shape[0], A_slm.shape[1], xoffA, yoffA) # A display modulates E_y in our setup
#
#     holo_x = c_phase2gray_interp(c_holo_x, gray_level_lut, phase_lut)
#     holo_y = c_phase2gray_interp(c_holo_y, gray_level_lut, phase_lut)
#
#     holo_x_displayed = slice_offset_part(holo_x, B_slm.shape[0], B_slm.shape[1], xoffB, yoffB) # B display modulates E_x in our setup
#     holo_y_displayed = slice_offset_part(holo_y, A_slm.shape[0], A_slm.shape[1], xoffA, yoffA) # A display modulates E_y in our setup
#
#     if not use_A_display:
#         holo_y_displayed = np.zeros_like(holo_y_displayed, dtype = np.uint8)
#     if not use_B_display:
#         holo_x_displayed = np.zeros_like(holo_x_displayed, dtype = np.uint8)
#
#     if double_pass:
#         c_holo_SLM = np.concatenate((c_holo_y_displayed,
#                                     np.flip(np.flip(c_holo_x_displayed, axis = 0), axis = 1)), # Flip due to telescope in our setup
#                                     axis = concat_ax) #  Concatenate A display (E_y) with B display (E_x)
#         holo_SLM = np.concatenate((holo_y_displayed,
#                                   np.flip(np.flip(holo_x_displayed, axis = 0), axis = 1)), # Flip due to telescope in our setup
#                                   axis = concat_ax) #  Concatenate A display (E_y) with B display (E_x)
#     else:
#         holo_SLM = slice_offset_part(holo_x_displayed, rowsSLM, colsSLM, xoffA, yoffA)
#
#     # Apply SLM corrections
#     if slm_correction_type == 'Direct Correction':
#         correction_map = plt.imread(slm_correction_filepath)
#         correction_map_phase = pad2size(correction_map, rowsSLM, colsSLM) / correction_map.max() * 2 * np.pi
#         c_holo_SLM = c_holo_SLM * np.exp(1j * correction_map_phase)
#         c_holo_SLM[np.abs(c_holo_SLM) == 0] = 0  # To avoid -0, +0 phase ambiguity
#         holo_SLM = c_phase2gray_interp(c_holo_SLM, gray_level_lut, phase_lut)
#
#     if slm_correction_type == 'Double-Pass Correction':
#         correction_map_phase = generate_double_pass_correction_map(slm_correction_filepath, rowsSLM, colsSLM,
#                                                                    gray_level_2pi, pupil_radius_pixel,
#                                                                    xoffA, yoffA, xoffB, yoffB)
#
#         c_holo_SLM = c_holo_SLM * np.exp(1j * correction_map_phase)
#         c_holo_SLM[np.abs(c_holo_SLM) == 0] = 0  # To avoid -0, +0 phase ambiguity
#         holo_SLM = c_phase2gray_interp(c_holo_SLM, gray_level_lut, phase_lut)
#
#     if modulation_type == 'Uniform Gray':
#         holo_SLM = np.uint8(np.ones([rowsSLM, colsSLM]) * coeff)
#
#     holo_SLM = np.uint8(holo_SLM)
#
#     return holo_SLM, np.float64(phase_lut/np.pi), np.float64(gray_level_lut)
#
#
# #                                     0                       1                     2             3                    4                      5                    6
# modulations_double_pass = ['Azimuthal Polarization', 'Radial Polarization', 'Knife Edge X', 'Knife Edge Y', 'Binary Mask Arrizon', 'Arrizon Test Pattern', 'Zeros Hologram']
# modulations_single_pass = ['Laguerre-Gaussian',      'Sinus Hologram',      'Arrizon Test', 'Arrizon Constant']
# modulation_type = modulations_double_pass[1]
# double_pass = 1
# rowsSLM, colsSLM = 1024, 1280
# laser_wavelength_nm = 488
# lin_period_X, lin_period_Y = 0, 0 #Due to Sampling issues of function compute_effective_field() for simulating we are limited to periods bigger than 4
# pupil_diameter_mm = 30
# coeff = 0
# xoffA, yoffA, xoffB, yoffB = 0, 0, 0, 0
# obl_ast_coeff, vert_ast_coeff = 0, 0
# use_A_display, use_B_display = True, True
# slm_correction_type = "No Correction"
# slm_correction_filepath = r'C:\Users\Usuario\OneDrive - Universitat de Barcelona\PhD\Code\SLM_Control\Vector_Beams_LabVIEW\CAL_LSH0804783_490nm.bmp'
# ampl_coeff = 0
# verbose = True
#
# holo_SLM, _, _ = compute_hologram(modulation_type, double_pass, rowsSLM, colsSLM, laser_wavelength_nm,
#                                  lin_period_X, lin_period_Y,
#                                  pupil_diameter_mm, obl_ast_coeff, vert_ast_coeff,
#                                  xoffA, yoffA, xoffB, yoffB,
#                                  use_A_display, use_B_display,
#                                  slm_correction_type, slm_correction_filepath,
#                                  coeff,
#                                  verbose=False)
#
#
#
#
#
