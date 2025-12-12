import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import time
from pyVeReady.misc.cin_final import *
import os
from dataclasses import dataclass, field
from pyVeReady.utils.image_utils import *
from skimage.transform import resize, rescale


@dataclass
class SLMLut:
    """
    A class to compute and store the SLM (Spatial Light Modulator) lookup table (LUT).
    It computes its associated full complex Arrizon accessible values.

    Attributes:
        gray_level_zero (int): The starting gray level for the LUT.
        gray_level_two_pi_span (int): The gray level span corresponding to a 2π phase shift.

        gray_level_lut (np.ndarray): Array of gray levels.
        phase_lut (np.ndarray): Array of phase values corresponding to gray levels.
        c_phase_lut (np.ndarray): Array of complex phase values corresponding to gray levels.
        m1_gray_lut (np.ndarray): Gray levels for the first modulation (m1).
        m2_gray_lut (np.ndarray): Gray levels for the second modulation (m2).
        m1_phase_lut (np.ndarray): Phase values for the first modulation (m1).
        m2_phase_lut (np.ndarray): Phase values for the second modulation (m2).
        c_m1_phase_lut (np.ndarray): Complex phase values for the first modulation (m1).
        c_m2_phase_lut (np.ndarray): Complex phase values for the second modulation (m2).
        full_complex_arrizon_lut (np.ndarray): Full complex accessible LUT using Arrizon encoding.
    Notes:
        Arrizon Full Complex Modulation LUT corresponds to all combinations between accessible values in the SLM.
        Having two accessible modulations c_m1 and c_m2 (represented in the complex plane), the resulting modulation
        with Arrizon encoding corresponds to (c_m1 + c_m2)/2.
    """
    gray_level_zero: int
    gray_level_two_pi_span: int

    def __post_init__(self):
        """
        Compute the LUTs after the object is initialized.
        """
        gray_level_zero = self.gray_level_zero
        gray_level_two_pi_span = self.gray_level_two_pi_span

        gray_level_lut = np.arange(gray_level_zero, gray_level_two_pi_span)
        phase_lut = (2 * np.pi / gray_level_two_pi_span) * (gray_level_lut - gray_level_zero)
        c_phase_lut = np.exp(1j * phase_lut)

        idx_m1_lut, idx_m2_lut = np.triu_indices(len(c_phase_lut))  # All combinations of indexes without repetition
        m1_gray_lut, m2_gray_lut = gray_level_lut[idx_m1_lut], gray_level_lut[idx_m2_lut]
        m1_phase_lut, m2_phase_lut = phase_lut[idx_m1_lut], phase_lut[idx_m2_lut]
        c_m1_phase_lut, c_m2_phase_lut = c_phase_lut[idx_m1_lut], c_phase_lut[idx_m2_lut]

        full_complex_arrizon_lut = (c_m1_phase_lut + c_m2_phase_lut) / 2

        # Save LUTs as class attributes
        self.gray_level_lut = gray_level_lut
        self.phase_lut, self.c_phase_lut = phase_lut, c_phase_lut
        self.m1_gray_lut, self.m2_gray_lut = m1_gray_lut, m2_gray_lut
        self.m1_phase_lut, self.m2_phase_lut = m1_phase_lut, m2_phase_lut
        self.c_m1_phase_lut, self.c_m2_phase_lut = c_m1_phase_lut, c_m2_phase_lut
        self.full_complex_arrizon_lut = full_complex_arrizon_lut

    def show_luts(self, show_phase = True, show_arrizon = True):
        if show_phase:
            plt.figure('Phase LUT')
            plt.plot(self.gray_level_lut, self.phase_lut / np.pi, color = 'blue')
            plt.xlabel('Gray Level')
            plt.ylabel(r'Phase Modulation ($\pi$ rad)')

        if show_arrizon:
            fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
            fig.canvas.manager.set_window_title('Arrizon LUT')
            ax.scatter(np.angle(self.full_complex_arrizon_lut), np.abs(self.full_complex_arrizon_lut), color = 'k', s = 0.1)
            ax.scatter(self.phase_lut, np.ones_like(self.phase_lut), color = 'b', s = 5)

@dataclass
class ModulationManager:
    # Optical and SLM parameters
    rowsSLM: int
    colsSLM: int
    double_pass: bool
    gray_level_zero: int
    gray_level_two_pi_span: int
    eff_pixel_size: int
    slm_pixel_size_um: float
    laser_wavelength_nm: float
    pupil_diameter_mm: float

    def __post_init__(self):
        # Generate SLM LUT
        self.slm_lut = SLMLut(self.gray_level_zero, self.gray_level_two_pi_span)

        # Setup Displays dimensions and Coordinates definition
        self.slm_canvas = np.zeros([self.rowsSLM, self.colsSLM])  # Full SLM
        if self.double_pass:
            self.A_slm, self.B_slm, self.concat_ax = split_array(self.slm_canvas)
            self.X, self.Y, self.R, self.phi = pixel_coordinates(np.repeat(np.repeat(self.A_slm, 2, axis=0), 2, axis=1))
        else:
            self.X, self.Y, self.R, self.phi = pixel_coordinates(np.repeat(np.repeat(self.slm_canvas, 2, axis=0), 2, axis=1))

        # Define Arrizon Cells
        self.m_pixels = int(self.eff_pixel_size) if self.eff_pixel_size >= 1 else 1
        self.X_arr, self.Y_arr = self.X[::2 * self.m_pixels, ::2 * self.m_pixels], self.Y[::2 * self.m_pixels, ::2 * self.m_pixels]
        self.R_arr, self.phi_arr = np.sqrt(self.X_arr ** 2 + self.Y_arr ** 2), np.arctan2(self.Y_arr, self.X_arr)

        # Generate Space for the Holograms
        self.c_holo_x = np.zeros_like(self.X, dtype=np.complex128)
        self.c_holo_y = np.zeros_like(self.Y, dtype=np.complex128)

    def generate_beam_shaping_hologram(self, modulation_type, microscope_correction=True, correct_gaussian_decay=False):
        # Compute initial beam modulation
        ex, ey, encoding = self.compute_beam_shaping_modulation(modulation_type)

        # Apply microscope transmission matrix correction
        if microscope_correction:
            ex_tmp = np.copy(ex)
            ex = (0.89914647+0.00036862j) * ex_tmp + (0.01320823+0.22222381j) * ey
            ey = (0.00741502-0.25135398j) * ex_tmp + (-0.90686885-0.0056573j) * ey

        # Compensate for Gaussian beam envelope
        if correct_gaussian_decay:
            beam_w_pixels = 205  # Obtained experimentally by trial and error
            edge_radius_pixels = int(1000 * (self.pupil_diameter_mm / 2) / self.slm_pixel_size_um)
            gaussian_beam = np.exp(-(self.X_arr ** 2 + self.Y_arr ** 2) / (beam_w_pixels ** 2))
            side_value = np.exp(-(edge_radius_pixels ** 2) / (beam_w_pixels ** 2))
            flattening_mask = side_value / gaussian_beam
            flattening_mask[flattening_mask > 1] = 0
            ex *= flattening_mask
            ey *= flattening_mask

        ex, ey = np.complex128(ex), np.complex128(ey)

        # Encode using Arrizon
        if encoding == 'arrizon':
            _, c_holo_x = compute_arrizon_holograms(ex,
                                                    self.slm_lut.m1_phase_lut,
                                                    self.slm_lut.m2_phase_lut,
                                                    self.slm_lut.m1_gray_lut,
                                                    self.slm_lut.m2_gray_lut,
                                                    self.slm_lut.full_complex_arrizon_lut,
                                                    self.m_pixels)

            _, c_holo_y = compute_arrizon_holograms(ey,
                                                    self.slm_lut.m1_phase_lut,
                                                    self.slm_lut.m2_phase_lut,
                                                    self.slm_lut.m1_gray_lut,
                                                    self.slm_lut.m2_gray_lut,
                                                    self.slm_lut.full_complex_arrizon_lut,
                                                    self.m_pixels)

            if c_holo_x.shape > self.X.shape:
                c_holo_x = crop_array_centered(c_holo_x, 0, 0, self.X.shape)
                c_holo_y = crop_array_centered(c_holo_y, 0, 0, self.X.shape)

            self.c_holo_x = c_holo_x
            self.c_holo_y = c_holo_y


    def add_linear_phase(self, lin_period_x, lin_period_y):
        """
        Adds linear phase modulation to the hologram.
        """
        c_lin_x = np.exp(1j * (2 * np.pi / lin_period_x) * self.X) if lin_period_x not in [0, np.inf] else 1
        c_lin_y = np.exp(1j * (2 * np.pi / lin_period_y) * self.Y) if lin_period_y not in [0, np.inf] else 1

        self.c_holo_x *= c_lin_x * c_lin_y
        self.c_holo_y *= c_lin_x * c_lin_y

    def add_spherical_wavefront(self, target_focal_length_mm):
        """
        Adds defocus phase using lens equation.
        """
        slm_pixel_size = self.slm_pixel_size_um * 1e-6  # Convert to meters
        wavelength = self.laser_wavelength_nm * 1e-9
        target_focal_length = target_focal_length_mm * 1e-3

        radial_coordinate_squared = (self.X ** 2 + self.Y ** 2) * slm_pixel_size ** 2
        wavefront = np.sqrt(target_focal_length ** 2 - radial_coordinate_squared) - target_focal_length
        wavefront = np.abs(wavefront) + 1e-16  # Add extra epsilon to prevent a discontinuity at the center (0,2pi equivalence)

        k_0 = 2*np.pi / wavelength
        c_spherical_wavefront = np.exp(-1j * k_0 * wavefront) if target_focal_length_mm not in [0, np.inf] else 1

        self.c_holo_x *= c_spherical_wavefront
        self.c_holo_y *= c_spherical_wavefront

    def add_astigmatism_phase(self, obl_ast_coeff):
        """
        Adds Seidel astigmatism phase term.
        """
        pupil_radius_pixel = (self.pupil_diameter_mm / 2) / (self.slm_pixel_size_um * 1e-3)
        R_norm = np.sqrt((self.X / pupil_radius_pixel) ** 2 + (self.Y / pupil_radius_pixel) ** 2)

        if obl_ast_coeff != 0:
            wavelength_nm = self.laser_wavelength_nm
            c_ast = np.exp(1j * (2 * np.pi / wavelength_nm) *
                           (obl_ast_coeff * wavelength_nm) *
                           R_norm ** 2 * np.cos(self.phi) ** 2)
        else:
            c_ast = 1

        self.c_holo_x *= c_ast
        self.c_holo_y *= c_ast

    def add_pupil_mask(self):
        """
        Masks the hologram with a circular pupil.
        """
        pupil_radius_pixel = (self.pupil_diameter_mm / 2) / (self.slm_pixel_size_um * 1e-3)
        pupil_mask = self.R ** 2 <= pupil_radius_pixel ** 2

        self.c_holo_x *= pupil_mask
        self.c_holo_y *= pupil_mask

    def compute_beam_shaping_modulation(self, modulation_type):
        """
        Returns ex, ey modulation arrays and encoding type for a given beam shaping type.
        """
        encoding = 'arrizon'

        match modulation_type:
            case 'Pure X':
                ex = np.ones_like(self.phi_arr)
                ey = np.zeros_like(self.phi_arr)
            case 'Pure Y':
                ex = np.zeros_like(self.phi_arr)
                ey = np.ones_like(self.phi_arr)
            case 'Azimuthal Polarization':
                ex, ey = -np.sin(self.phi_arr), np.cos(self.phi_arr)
            case 'Radial Polarization':
                ex, ey = np.cos(self.phi_arr), np.sin(self.phi_arr)
                #  ex *= self.R_arr / R0
                #  ey *= self.R_arr / R0
            case 'Circular Left':
                ex = np.ones_like(self.phi_arr)
                ey = ex * 1j
            case 'Circular Right':
                ex = np.ones_like(self.phi_arr)
                ey = ex * -1j
                #  ex *= self.R_arr / R0
                #  ey *= self.R_arr / R0
            case 'Laguerre-Gaussian Lin':
                ex = ey = np.exp(1j * self.phi_arr)
            case 'Laguerre-Gaussian Left':
                ex = np.exp(1j * self.phi_arr)
                ey = ex * 1j
            case 'Laguerre-Gaussian Right':
                ex = np.exp(1j * self.phi_arr)
                ey = ex * -1j
            case 'Knife Edge X (Phase)':
                ex = ey = self.X_arr < 0
            case 'Knife Edge Y (Phase)':
                ex = ey = self.Y_arr < 0
            case 'Zeros Hologram':
                ex = ey = np.ones_like(self.phi_arr) #ex = ey = np.zeros_like(self.phi_arr)
            case 'Sinus Hologram':
                sinus = np.exp(1j * (np.pi * np.sin(self.X_arr / 4)))
                ex = ey = sinus
            case 'Artur Beam':
                path = r'C:\Users\laboratori\Desktop\PySLM_VeReady\pyVeReady\misc\Artur2.0\coeficients_NA=0.95_coef=6_NP=512_nterms=8_zonaVisPp2=6limit=1.npy' #path_to_artur_beam.npy' #Replace with actual path
                n_points = int(1000 * (self.pupil_diameter_mm / self.slm_pixel_size_um) / (2 * self.m_pixels))
                ex, ey = load_artur_beam(path, n_points)
                ex = pad2size(ex, *self.phi_arr.shape)
                ey = pad2size(ey, *self.phi_arr.shape)
            case _:
                raise ValueError(f"Unknown modulation type: {modulation_type}")

        return ex, ey, encoding



def get_holo_openCL(C1, C_SLM1, verbose = 0):
    """ From David Maluenda Niubó, PyHolo Repository:
        https://github.com/dmaluenda/pyHolo

        It found the nearest complex value from the desired C1 to all accessible C_SLM1 values.
        Needs to have the file mapa_holo_kernel.cl in the same directory as the script.

        params:
          - C1 : 2D-complex array of the desired values (target hologram)
          - C_SLM1 : 1D-complex array containing all accessible values by the SLM
          - verbose: verbose level.

        return: A 2D-integer array of same size/shape of C1 with
                every nearest value indix according to the C_SLM1 order
    """

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'  # Just to print compiling info

    # Target hologram
    N = C1.shape  # To reshape-back
    desired_flat = C1.flatten()  # We will work on a flattened arrays
    desired_real = desired_flat.real.astype(np.float32)
    desired_imag = desired_flat.imag.astype(np.float32)

    # Accessible values by the SLM
    acc_real = C_SLM1.real.astype(np.float32)
    acc_imag = C_SLM1.imag.astype(np.float32)
    m = acc_imag.size

    if verbose > 1:
        print(f"Total number of accessible values (m): {m}")

    # Resulting array
    slm_flat = np.zeros_like(desired_real, dtype='float32')

    # Setting up openCL
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    dr_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                       hostbuf=desired_real)
    di_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                       hostbuf=desired_imag)
    ar_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                       hostbuf=acc_real)
    ai_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                       hostbuf=acc_imag)
    slm_buf = cl.Buffer(ctx, mf.WRITE_ONLY, slm_flat.nbytes)

    # Read the kernel.cl file
    root = os.path.abspath(os.path.dirname(__file__))
    with open(root + r'\mapa_holo_kernel.cl', 'r') as file:
        openCL_code = file.read()

    # Compile the code
    t0 = time.time()
    prg = cl.Program(ctx, openCL_code).build()
    t1 = time.time()

    # Running the openCL program
    prg.nearest(queue, slm_flat.shape, None,
                np.uint16(m), dr_buf, di_buf, ar_buf, ai_buf, slm_buf)
    t2 = time.time()

    # setting up the output
    res_slm = np.empty_like(slm_flat)  # just a buffer
    t3 = time.time()

    # Transfering data
    cl.enqueue_copy(queue, res_slm, slm_buf)
    t4 = time.time()

    if verbose > 1:
        print(" --- ")
        print(f"cl.Program:  {t1 - t0:.2f}")
        print(f"prg.nearest: {t2 - t1:.2f}")
        print(f"np.empty_lk: {t3 - t2:.2f}")
        print(f"cl.enque_cp: {t4 - t3:.2f}")

    # reshaping-back
    p1 = res_slm.reshape(N).astype('int')

    return p1  # 2D-integer array made of the indices of every nearest value

def compute_arrizon_holograms(c_modulation, m1_phase_lut, m2_phase_lut, m1_gray_lut, m2_gray_lut, C_SLM_lut, m_pixels, method ='OpenCL'):
    """
    Computes holograms based on Arrizon's method for a given input field. This method encodes full complex modulation
    by constructing a macro pixel made of symmetric complex modulations c_m1 and c_m2 [[c_m1, c_m2],[c_m2, c_m1]].
    In which the resulting modulation is (c_m1 + c_m2)/2.

    Parameters:
        c_modulation (numpy.ndarray): The complex 2D input complex modulation for hologram computation.
        m1_phase_lut (numpy.ndarray): Phase lookup table for the first sub-macro pixel modulation.
        m2_phase_lut (numpy.ndarray): Phase lookup table for the second sub-macro piexl modulation.
        m1_gray_lut (numpy.ndarray): Gray level lookup table for the first sub-macro pixel modulation.
        m2_gray_lut (numpy.ndarray): Gray level lookup table for the second sub-macro pixel modulation.
        m_pixels (int): Number of sub-macro pixel modulations.
        C_SLM_lut (numpy.ndarray): Lookup table of the SLM's complex response.
        method (str, optional): The method to compute the closest indices in the lookup table.
          Options are:
          - 'OpenCL': Use OpenCL acceleration for hologram computation (default).
          - Other: Fallback to a slower CPU-based computation.

    Returns:
        holo (numpy.ndarray): The amplitude hologram (2D uint8 array).
        c_holo (numpy.ndarray): The complex hologram (2D complex128 array).

    Notes:
        The output holograms are 2x the size of the input field in both dimensions.
        The `holo` array contains gray levels from the LUTs (`m1_gray_lut`, `m2_gray_lut`).
        The `c_holo` array contains the phase components as complex exponential values
        derived from the phase LUTs (`m1_phase_lut`, `m2_phase_lut`).
    """
    if method == 'OpenCL':
        idx_closest_2D = get_holo_openCL(c_modulation, C_SLM_lut)  # 2D array of nearest indices
    else:
        idx_closest_2D = np.zeros(c_modulation.shape, dtype=np.uint64)
        for ii in range(c_modulation.shape[0]):
            for jj in range(c_modulation.shape[1]):
                idx_closest_2D[ii, jj] = np.argmin(abs(c_modulation[ii, jj] - C_SLM_lut))

    m1_gray_level = m1_gray_lut[idx_closest_2D].reshape(c_modulation.shape)
    m2_gray_level = m2_gray_lut[idx_closest_2D].reshape(c_modulation.shape)
    c_m1 = np.exp(1j * m1_phase_lut[idx_closest_2D])
    c_m2 = np.exp(1j * m2_phase_lut[idx_closest_2D])

    holo = generate_checkerboard(m1_gray_level, m2_gray_level, m_pixels)
    c_holo = generate_checkerboard(c_m1, c_m2, m_pixels)

    return holo, c_holo


def c_phase2gray_interp(c_array, gray_lut, phase_lut):
    """
    Interpolates the phase of complex values to grayscale values given a
    gray-phase lookup table relation. Requires a linear phase lookup table.

    Parameters:
        c_array : numpy.ndarray
            Array of complex numbers.
        gray_lut : numpy.ndarray
            Grayscale lookup table for phase-to-intensity mapping.
        phase_lut : numpy.ndarray
            Phase lookup table, should be monotonically increasing in [0, 2π].

    Returns:
        numpy.ndarray
            Array of grayscale intensities corresponding to the phases.
    """
    phase = np.angle(c_array)  # Between [-pi, +pi]
    phase = phase + 2 * np.pi * (np.sign(phase) < 0)  # Bounded to [0, 2pi]

    holo = np.interp(phase, phase_lut, gray_lut)
    return holo

def generate_double_pass_correction_map(slm_correction_filepath, rowsSLM, colsSLM,
                                        GRAY_LEVEL_2PI, pupil_radius_pixel,
                                        xoffA, yoffA, xoffB, yoffB):
    """
    Generate the double pass correction map in phase[rad] given the SLM correction map and the beam
    locations on the sub-displays A and B.

    Parameters:
        slm_correction_filepath (str): Path to the SLM correction file.
        rowsSLM (int): Number of rows for the SLM array.
        colsSLM (int): Number of columns for the SLM array.
        GRAY_LEVEL_2_PI (float): Scaling factor to normalize the correction map.
        pupil_radius_pixel (float): Radius of the pupil mask in pixels.
        xoffA, yoffA (int): Offsets for region A.
        xoffB, yoffB (int): Offsets for region B.

    Returns:
        np.ndarray: Phase correction map (probably exceeds 2pi).
    """

    # Load the SLM correction map
    correction_map = plt.imread(slm_correction_filepath)
    correction_map_phase = pad2size(correction_map, rowsSLM, colsSLM) / correction_map.max() * 2 * np.pi

    # Split the correction map into A and B displays
    A_correction, B_correction, concat_ax = split_array(correction_map_phase)

    # Generate pixel coordinates and masks for spots in A and B regions
    X_A, Y_A, _, _ = pixel_coordinates(A_correction, -xoffA, -yoffA)
    X_B, Y_B, _, _ = pixel_coordinates(B_correction, xoffB, yoffB)
    spot_in_A_mask = (X_A ** 2 + Y_A ** 2) <= pupil_radius_pixel ** 2
    spot_in_B_mask = (X_B ** 2 + Y_B ** 2) <= pupil_radius_pixel ** 2

    # Flip masks and correction maps
    flipped_A_correction = np.flip(np.flip(A_correction, axis=0), axis=1)
    flipped_B_correction = np.flip(np.flip(B_correction, axis=0), axis=1)
    flipped_mask_spot_in_A = np.flip(np.flip(spot_in_A_mask, axis=0), axis=1)
    flipped_mask_spot_in_B = np.flip(np.flip(spot_in_B_mask, axis=0), axis=1)

    # Apply corrections
    A_total_correction = A_correction * spot_in_A_mask
    B_total_correction = B_correction * spot_in_B_mask
    A_total_correction[spot_in_A_mask] += flipped_B_correction[flipped_mask_spot_in_B]
    B_total_correction[spot_in_B_mask] += flipped_A_correction[flipped_mask_spot_in_A]

    holo_correction_phase = np.concatenate((A_total_correction, B_total_correction), concat_ax)  # Phase Correction

    return holo_correction_phase

def compute_c_linear_phase(method, lin_period_x, lin_period_y, X, Y, macropixel_size = 2):
    """
    Compute the linear phase components along the x and y axes based on the specified method.

    Parameters:
        method (str): The method to compute the linear phase. Options are "classic" or "macro-pixel".
        lin_period_x (float): Linear phase period along the x-axis. If 0 or np.inf, no phase is applied along x.
        lin_period_y (float): Linear phase period along the y-axis. If 0 or np.inf, no phase is applied along y.
        X (ndarray): 2D array representing the x-coordinates.
        Y (ndarray): 2D array representing the y-coordinates.

    Returns:
        tuple: A tuple (c_lin_phase_x, c_lin_phase_y) where:
            - c_lin_phase_x (ndarray): Complex array of the linear phase along the x-axis.
            - c_lin_phase_y (ndarray): Complex array of the linear phase along the y-axis.
    """
    if method == "classic":
        c_lin_phase_x = np.exp(1j * (2 * np.pi / lin_period_x) * X) if (lin_period_x != 0 and lin_period_x != np.inf) else 1
        c_lin_phase_y = np.exp(1j * (2 * np.pi / lin_period_y) * Y) if (lin_period_y != 0 and lin_period_y != np.inf) else 1
    elif method == "macro-pixel":
        X_downsampled, Y_downsampled = X[::macropixel_size, ::macropixel_size], Y[::macropixel_size, ::macropixel_size]
        c_lin_phase_x_downsampled = np.exp(1j * (2 * np.pi / lin_period_x) * X_downsampled) if (lin_period_x != 0 and lin_period_x != np.inf) else np.ones_like(X_downsampled)
        c_lin_phase_y_downsampled = np.exp(1j * (2 * np.pi / lin_period_y) * Y_downsampled) if (lin_period_y != 0 and lin_period_y != np.inf) else np.ones_like(Y_downsampled)

        c_lin_phase_x = upsample_array_macropixel(c_lin_phase_x_downsampled, macropixel_size)
        c_lin_phase_y = upsample_array_macropixel(c_lin_phase_y_downsampled, macropixel_size)

        if c_lin_phase_x.shape[0] > X.shape[0] or c_lin_phase_x.shape[1] > X.shape[1]:
            c_lin_phase_x = crop_array_centered(c_lin_phase_x, 0, 0, X.shape)
            c_lin_phase_y = crop_array_centered(c_lin_phase_y, 0, 0, X.shape)

    else:
        raise ValueError("Unknown linear phase method")

    return c_lin_phase_x, c_lin_phase_y

def compute_arrizon_coordinates(X, Y, m_pixels = 2):
    """
    Compute downsampled Cartesian coordinates and their polar equivalents using Arrizon encoding.

    This function performs coordinate downsampling and conversion to polar coordinates, typically used
    in spatial light modulator (SLM) applications for complex field encoding. The downsampling follows
    the Arrizon method of pairwise combination.

    Parameters:
        X (np.ndarray): 2D array of x-coordinates from a meshgrid
        Y (np.ndarray): 2D array of y-coordinates from a meshgrid (same shape as X)
        m_pixels (int, optional): Downsampling factor for pixel selection. Defaults to 2 (every other pixel).

    Returns:
        tuple: Four 2D numpy arrays containing:
            - X_arr (np.ndarray): Downsampled x-coordinates
            - Y_arr (np.ndarray): Downsampled y-coordinates
            - R_arr (np.ndarray): Radial coordinates (sqrt(X_arr² + Y_arr²)
            - phi_arr (np.ndarray): Angular coordinates in radians (arctan2(Y_arr, X_arr))
    """

    X_arr, Y_arr = X[::m_pixels, ::m_pixels], Y[::m_pixels, ::m_pixels]  # Arrizon Coordinates
    R_arr, phi_arr = np.sqrt(X_arr ** 2 + Y_arr ** 2), np.arctan2(Y_arr, X_arr)

    return X_arr, Y_arr, R_arr, phi_arr

def upsample_array_macropixel(array, macropixel_size):
    """
    Reverts the effect of subsampling by creating blocks of size macropixel_size x macropixel_size with the same value.
    Assumes that array was downsampled using X[::macropixel_size, ::macropixel_size].
    """
    return np.kron(array, np.ones((macropixel_size, macropixel_size)))


def generate_checkerboard(m1, m2, m_pixels):
    """
    Generates a checkerboard pattern where each macro pixel (2*m_pixels x 2*m_pixels)
    contains four sub-pixels (m_pixels x m_pixels each). The top-left and bottom-right sub-pixels
    use values from m1, while the top-right and bottom-left sub-pixels use values from m2.

    Parameters:
        m1 (2D array): Values for the diagonal sub-pixels.
        m2 (2D array): Values for the counter-diagonal sub-pixels.
        m_pixels (int): Size of each subpixel block.

    Returns:
        numpy.ndarray: The generated checkerboard pattern.
    """
    assert m1.shape == m2.shape, "m1 and m2 must have the same shape"

    macropixels_rows, macropixels_cols = m1.shape
    output_rows = macropixels_rows * 2 * m_pixels
    output_cols = macropixels_cols * 2 * m_pixels

    # Initialize empty array
    output = np.zeros((output_rows, output_cols), dtype=m1.dtype)

    for row in range(macropixels_rows):
        for col in range(macropixels_cols):
            val1 = m1[row, col]
            val2 = m2[row, col]

            start_row, start_col = row * 2 * m_pixels, col * 2 * m_pixels

            # Fill the subpixel regions
            output[start_row:start_row + m_pixels, start_col:start_col + m_pixels] = val1  # Top-left
            output[start_row + m_pixels:start_row + 2 * m_pixels, start_col + m_pixels:start_col + 2 * m_pixels] = val1  # Bottom-right
            output[start_row:start_row + m_pixels, start_col + m_pixels:start_col + 2 * m_pixels] = val2  # Top-right
            output[start_row + m_pixels:start_row + 2 * m_pixels, start_col:start_col + m_pixels] = val2  # Bottom-left

    return output

def compute_displacement(FOV, n_steps, focal_length, central_period_pix, wavelength, slm_pix_size):
    postion_0 = wavelength * focal_length / (central_period_pix * slm_pix_size)
    pos_max = postion_0 + FOV / 2
    pos_min = postion_0 - FOV / 2
    positions = np.linspace(pos_min, pos_max, n_steps)
    periods = wavelength * focal_length / (positions * slm_pix_size)
    return positions, periods

def deflection_angle_to_period(angle, wavelength, pixel_size_slm):
    """
    Converts a deflection angle to the corresponding linear phase period.
    If deflection angle is 0 it returns a period of np.inf

    Parameters:
        angle (float or array-like): Deflection angle(s) in radians.
        wavelength (float): Wavelength of the incident light.
        pixel_size_slm (float): Pixel size of the spatial light modulator (SLM).

    Returns:
        numpy.ndarray: Grating period(s).
    """
    angle = np.array(angle, dtype = np.float64)  # Convert to NumPy array if not already
    period = np.full_like(angle, np.inf)  # Initialize with np.inf

    mask = angle != 0  # Boolean mask for nonzero angles
    period[mask] = wavelength / (angle[mask] * pixel_size_slm)  # Compute only where angle ≠ 0
    return period

def period_to_deflection_angle(period, wavelength, pixel_size_slm):
    """
     Converts a linear phase period to the corresponding deflection angle.
     If period is 0 it returns a deflection angle of 0 radians.

    Parameters:
        period (float or array-like): Grating period(s).
        wavelength (float): Wavelength of the incident light.
        pixel_size_slm (float): Pixel size of the spatial light modulator (SLM).

    Returns:
        numpy.ndarray: Deflection angle(s).
    """
    period = np.array(period, dtype = np.float64)  # Convert to NumPy array if not already
    angle = np.zeros_like(period)  # Initialize array with zero

    mask = period != 0  # Boolean mask for nonzero periods
    angle[mask] = wavelength / (period[mask] * pixel_size_slm)  # Compute only where period ≠ 0
    return angle