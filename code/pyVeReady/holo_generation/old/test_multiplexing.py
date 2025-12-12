from pyVeReady.holo_generation.old.gerchberg_saxton import *
from pyVeReady.holo_generation.holo_generation_functions import *
from pyVeReady.utils.image_utils import *
matplotlib.use('tkagg')
np.random.seed(1)
import slmpy


matplotlib.use('TkAgg')

rows, cols = 1024, 640

target_img_test = create_ub_image(620, 620, 60, row_spacing=15, col_spacing=15)  # Adjust row and column spacing here
target_img_test = np.ones((400,400))

# Gerchberg-Saxton Parameters
holo_padded_size_factor_test = 1
noise_area_factor_test = 0
initial_phase_test = 'Quadratic'
use_noise_area_test = False
n_iterations_test = 50

c_holo_effective = compute_gerchberg_saxton_hologram(target_img_test, initial_phase_test, n_iterations_test,
                                                     holo_padded_size_factor_test, use_noise_area_test,
                                                     noise_area_factor_test)

X, Y, R, phi = pixel_coordinates(c_holo_effective)
# c_holo_effective = c_holo_effective #* np.exp(1j * phi) * np.exp(1j * 2*np.pi * X/10)


# Compute SLM LUTs
gray_level_0 = 0
gray_level_2pi = 195
SLM_LUT = SLMLut(gray_level_0, gray_level_2pi)

gray_level_lut = SLM_LUT.gray_level_lut
phase_lut = SLM_LUT.phase_lut
m1_gray_lut, m2_gray_lut = SLM_LUT.m1_gray_lut, SLM_LUT.m2_gray_lut
m1_phase_lut, m2_phase_lut = SLM_LUT.m1_phase_lut, SLM_LUT.m2_phase_lut
full_complex_arrizon_lut = SLM_LUT.full_complex_arrizon_lut

holo_effective = c_phase2gray_interp(c_holo_effective, gray_level_lut, phase_lut)


semi_display = pad2size(holo_effective, rows, cols)
zeros = np.zeros((rows, cols))
display = np.hstack((zeros, semi_display))

plt.figure()
plt.imshow(display)
plt.show()

slm = slmpy.SLMdisplay()
slm.updateArray(np.uint8(display))
