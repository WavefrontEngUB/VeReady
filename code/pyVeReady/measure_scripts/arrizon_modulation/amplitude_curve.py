from pyVeReady.utils.image_utils import *
from pyVeReady.holo_generation import *
import imagingcontrol4 as ic4
import slmpy
import pyautogui

if __name__ == '__main__':
    n_amplitudes = 26
    target_amplitudes = np.linspace(0, 1, n_amplitudes)

    modulation_type = 'Arrizon Constant'
    double_pass = True
    rowsSLM, colsSLM = 1024, 1280
    laser_wavelength_nm = 488

    lin_period_X, lin_period_Y = -30, 0
    pupil_diameter_mm = 30
    obl_ast_coeff = 0
    xoffA, yoffA, xoffB, yoffB = 0, 0, 0, 0
    use_A_display, use_B_display = True, False
    slm_correction_type, slm_correction_filepath = 'No Correction', None

    verbose = True

    n_subpixel_sizes = 10
    pixel_size_values = np.arange(1, n_subpixel_sizes + 1)

    inp = input('Move mouse to Snap and press enter')
    x, y = pyautogui.position()

    slm = slmpy.SLMdisplay()
    list_pixel_size = []
    for aa, m_pixels in enumerate(pixel_size_values):
        print(f'Pixels {m_pixels}')
        list_pixel_size.append(m_pixels)
        images_same_pixel_size = []
        for ii, coeff in enumerate(target_amplitudes):
            print(ii)

            # Compute Hologram
            [holo_SLM, SLM_LUT] = compute_hologram(modulation_type, double_pass, rowsSLM, colsSLM, laser_wavelength_nm,
                                                     lin_period_X, lin_period_Y,
                                                     pupil_diameter_mm, obl_ast_coeff, m_pixels,
                                                     xoffA, yoffA, xoffB, yoffB,
                                                     use_A_display, use_B_display,
                                                     slm_correction_type, slm_correction_filepath,
                                                     coeff,
                                                     verbose)
            slm.updateArray(holo_SLM)
            plt.imshow(holo_SLM, cmap='gray', vmin = 0, vmax = 255)
            time.sleep(0.5)

            # Take Image
            pyautogui.click(x, y)