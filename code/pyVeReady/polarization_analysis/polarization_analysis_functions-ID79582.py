import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pypolar
from scipy.ndimage import rotate
# from pyVeReady.utils.image_utils import *
from matplotlib_scalebar.scalebar import ScaleBar
import sys


def compute_stokes(data_list, characterization_method):
    if characterization_method == 'Classic':
        s0_data, s1_data, s2_data, s3_data = compute_stokes_classic(data_list)

    elif characterization_method == 'Rotating QWP':
        s0_data, s1_data, s2_data, s3_data = compute_stokes_rotating_qwp(data_list)
    else:
        raise ValueError('Invalid characterization method')
    return s0_data, s1_data, s2_data, s3_data

def compute_stokes_classic(data_list):
    """
    Coordinate System is defined looking towards the source.
    Measures must be taken in the following order:
        - No Pol, No QWP (S0).
        - Pol Horizontal.
        - Pol Vertical.
        - Pol +45 Degrees
        - Pol -45 Degrees
        - Pol 90 Degrees, QWP +45 Degrees (R)
        - Pol 90 Degrees, QWP -45 Degrees (L)
    """
    s0_data = data_list[0]
    s1_data = data_list[1] - data_list[2]
    s2_data = data_list[3] - data_list[4]
    s3_data = data_list[5] - data_list[6]

    return s0_data, s1_data, s2_data, s3_data

def compute_stokes_rotating_qwp(data_list):
    """
    Computes the Stokes parameters according to the rotating QWP method
    with 8 measurements algebraicly. Measurements need to be equally spaced
    from 0 to 180 degrees.

    - Params:  data_list (List[np.ndarray]): Polarization images with increasing QWP angle.
    - Returns : s_map with indexes [rows, cols, s_i]
    """
    n = len(data_list)
    qwp_angles = np.arange(n) * np.pi / n

    a_list = [2 / n * data for data in data_list]
    b_list = [4 / n * data * np.sin(2 * qwp_angle) for data, qwp_angle in zip(data_list, qwp_angles)]
    c_list = [4 / n * data * np.cos(4 * qwp_angle) for data, qwp_angle in zip(data_list, qwp_angles)]
    d_list = [4 / n * data * np.sin(4 * qwp_angle) for data, qwp_angle in zip(data_list, qwp_angles)]

    a, b, c, d = sum(a_list), sum(b_list), sum(c_list), sum(d_list)

    s0 = a - c
    s1 = 2 * c
    s2 = 2 * d
    s3 = b

    return s0, s1, s2, s3


def compute_stokes_from_labview_array(labview_array):
    """
    Computes the Stokes parameters from a LabVIEW-style 3D array using the rotating QWP method.

    This function assumes that the input array contains polarization intensity images
    taken at equally spaced quarter-wave plate (QWP) angles from 0 to 180 degrees.
    The computation follows the standard algebraic method using 8 or more measurements.

    Parameters:
        labview_array (np.ndarray): A 3D NumPy array of shape (N, H, W), where:
            - N is the number of QWP angle measurements (must be ≥ 4 and equally spaced from 0 to 180°).
            - H, W are the height and width of each 2D polarization image.

    Returns:
        stokes_labview_array (np.ndarray): A 3D array of shape (4, H, W), where:
            - Index 0 corresponds to S0
            - Index 1 corresponds to S1
            - Index 2 corresponds to S2
            - Index 3 corresponds to S3
    """
    labview_array = np.array(labview_array, dtype=np.float64)  # Ensure we deal with a 3D numpy array
    data_list_labview = [labview_array[ii] for ii in range(labview_array.shape[0])]  # Convert to a list of 2D arrays
    s0, s1, s2, s3 = compute_stokes_rotating_qwp(data_list_labview)  # Compute the Stokes Parameters
    stokes_labview_array = np.stack([s0, s1, s2, s3], axis=0)  # Arrange the Stokes parameters into a 3D array for LabVIEW
    return stokes_labview_array

def polarization_map_analysis(list_beam_vectors, vector_tpye,
                              show_figure = True,
                              show_relative_phase_map = True,
                              show_azimuthal_orientation_map = False,
                              show_circ_error = False,
                              orientation_into_first_quadrant = False,
                              draw_step = 20, arrow_size = 0.8,
                              filter_arrow_intensity = True,
                              figname=""):

    if vector_tpye == 'Stokes':  # Convert Stokes to Jones
        s0_data, s1_data, s2_data, s3_data = list_beam_vectors[0], list_beam_vectors[1], list_beam_vectors[2], list_beam_vectors[3]
        ex_data, ey_data = stokes2jones_pypol(s0_data, s1_data, s2_data, s3_data)

    elif vector_tpye == 'Jones':  # Convert Jones to Stokes
        ex_data, ey_data = list_beam_vectors[0], list_beam_vectors[1]
        s0_data, s1_data, s2_data, s3_data = jones2stokes_pypol(ex_data, ey_data)

    else:
        raise ValueError('Invalid Beam Vector type')

    relative_phase = np.angle(np.exp(1j * np.angle(ey_data)) / np.exp(1j * np.angle(ex_data)))
    maj_axes, min_axes, orientation_map = compute_ellipse_pypol([ex_data, ey_data],
                                                                parameter_type = 'Jones')
    error = (-np.pi / 2) - relative_phase

    if orientation_into_first_quadrant:
        mapped_orientation_map = map_angles_to_first_quadrant(orientation_map)
    else:
        mapped_orientation_map = orientation_map

    fig = None
    axes = None
    if show_figure:
        # Prepare Figure axes
        num_plots = 1 + int(show_relative_phase_map) + int(show_azimuthal_orientation_map) + int(show_circ_error)
        if num_plots > 1:
            fig, axes = plt.subplots(1, num_plots, sharex = True, sharey = True, figsize = (15, 6))
            ax_pol = axes[0]
        else:
            fig, axes = plt.subplots()
            ax_pol = axes

        if figname != "":
            fig.canvas.manager.set_window_title(figname)

        # Plot Polarisation Ellipses Map
        im_pol = ax_pol.imshow(s0_data, cmap = 'gray')
        ax_pol.set_title('S0')
        rows, cols = s0_data.shape
        for ii in range(rows // draw_step):
            for jj in range(cols // draw_step):
                maj_ax = maj_axes[draw_step * ii, draw_step * jj]
                min_ax = min_axes[draw_step * ii, draw_step * jj]
                orientation_angle = orientation_map[draw_step * ii, draw_step * jj]

                if filter_arrow_intensity:
                    if s0_data[draw_step * ii, draw_step * jj] > s0_data.min() + 0.2 * (s0_data.max() - s0_data.min()):
                    # For Having a Unit sized Ellipse Draw
                        maj_ax, min_ax = maj_ax / maj_ax, min_ax / maj_ax
                        e = Ellipse(xy=(draw_step * jj, draw_step * ii),
                                    width=draw_step * arrow_size * maj_ax,
                                    height=draw_step * arrow_size * min_ax,
                                    angle=-orientation_angle * 180 / np.pi, # This minus sign in the angle is due to the
                                    fill=False,                              # inversion of the angle that imshow performs
                                    color='cyan')
                                    #color = plt.cm.viridis(min_ax / maj_ax))   # when set_xlim() or set_ylim() are not used.
                        ax_pol.add_artist(e)
                else:
                    # For Having a Unit sized Ellipse Draw
                    maj_ax, min_ax = maj_ax / maj_ax, min_ax / maj_ax
                    e = Ellipse(xy=(draw_step * jj, draw_step * ii),
                                width=draw_step * arrow_size * maj_ax,
                                height=draw_step * arrow_size * min_ax,
                                angle=- orientation_angle * 180 / np.pi,  # This minus sign in the angle is due to the
                                fill=False,  # inversion of the angle that imshow performs
                                color=plt.cm.viridis(min_ax / maj_ax))  # when set_xlim() or set_ylim() are not used.
                    ax_pol.add_artist(e)

        fig.colorbar(im_pol, ax=ax_pol, shrink=0.6)
        ax_pol.set_xticks([])
        ax_pol.set_yticks([])
        scalebar = ScaleBar(6, "µm", frameon=False, color='white', location="lower right", height_fraction=.02)
        ax_pol.add_artist(scalebar)

        # Extra Plots
        flags = [show_relative_phase_map, show_azimuthal_orientation_map, show_circ_error]
        extra_plot_data = [relative_phase / np.pi, mapped_orientation_map / np.pi, error / np.pi]
        extra_plot_titles = [r'$\delta = \phi_y - \phi_x$ ($\pi$ rad)',
                             r'Mapped $\theta_{ell}$ ($\pi$ rad)' if orientation_into_first_quadrant else r'$\theta_{ell}$ ($\pi$ rad)',
                             r'$-\pi/2 - \delta$ ($\pi$ rad)']
        extra_plot_colormaps = ['twilight', 'gray', 'twilight']
        counter = 1
        for (boolean, data, title, clrmp) in zip(flags, extra_plot_data, extra_plot_titles, extra_plot_colormaps):
            if boolean:
                im = axes[counter].imshow(data, cmap = clrmp)
                axes[counter].set_title(title)
                fig.colorbar(im, ax = axes[counter], shrink = 0.6)
                counter += 1

        plt.show()
        plt.pause(0.001)
        fig.tight_layout()
    return fig, axes, ex_data, ey_data, relative_phase, mapped_orientation_map, error

def compute_ellipse_pypol(beam_parameters_list, parameter_type,
                          show = False):
    """
    Compute the ellipse parameters (major axis, minor axis, and orientation angle)
    from a list of beam vectors using the PyPol library. If Stokes vectors are input
    they are converted into its associated Jones vectors.

    Parameters:
        beam_parameters_list (list): Input beam vectors, either in Stokes or Jones representation.
        parameter_type (str): Type of the beam vectors, either 'Stokes' or 'Jones'.
        show (bool, optional): If True, visualize the ellipse for the first beam parameter. Default is False.

    Returns:
        tuple: Arrays representing the major axes, minor axes, and orientation angles of the ellipses.
    """
    if parameter_type == 'Stokes':
        s0_input, s1_input, s2_input, s3_input = beam_parameters_list[0], beam_parameters_list[1], beam_parameters_list[2], beam_parameters_list[3]
        ex_input, ey_input = stokes2jones_pypol(s0_input, s1_input, s2_input, s3_input)
    elif parameter_type == 'Jones':
        ex_input, ey_input = beam_parameters_list[0], beam_parameters_list[1]
    else:
        raise ValueError('Invalid Beam Vector type')

    ex_flat = ex_input.flatten()
    ey_flat = ey_input.flatten()

    orientation_angles_flat = np.zeros_like(ex_flat, dtype = np.float64)
    maj_axes_flat = np.zeros_like(ex_flat, dtype = np.float64)
    min_axes_flat = np.zeros_like(ex_flat, dtype = np.float64)
    for ii, (ex, ey) in enumerate(zip(ex_flat, ey_flat)):
        maj_ax, min_ax = pypolar.jones.ellipse_axes(np.array([ex, ey], dtype = np.complex128))
        orientation_angle = pypolar.jones.ellipse_azimuth(np.array([ex, ey], dtype = np.complex128))
        maj_axes_flat[ii] = maj_ax
        min_axes_flat[ii] = min_ax
        orientation_angles_flat[ii] = orientation_angle

    maj_axes = maj_axes_flat.reshape(ex_input.shape)
    min_axes = min_axes_flat.reshape(ex_input.shape)
    orientation_angles = orientation_angles_flat.reshape(ex_input.shape)

    if show:
        if len(ex_flat) == 1:
            pypolar.visualization.draw_jones_ellipse(np.array([ex_input, ey_input]))
        else:
            pypolar.visualization.draw_jones_ellipse(np.array([ex_input[0], ey_input[0]]))

    return maj_axes, min_axes, orientation_angles

def stokes2jones_pypol(s0_input, s1_input, s2_input, s3_input):
    """
    Convert Stokes parameters to Jones vector components (Ex and Ey) using PyPol.

    This function takes n-dimensional arrays of Stokes parameters (s0, s1, s2, s3)
    as input, flattens them, converts each set of Stokes parameters to Jones vector
    components (Ex and Ey) using the PyPol library, and reshapes the results back
    to the original input shape.

    Parameters:
        s0_input (ndarray): n-dimensional array of the S0 Stokes parameter.
        s1_input (ndarray): n-dimensional array of the S1 Stokes parameter.
        s2_input (ndarray): n-dimensional array of the S2 Stokes parameter.
        s3_input (ndarray): n-dimensional array of the S3 Stokes parameter.

    Returns:
        tuple: A tuple containing:
            - Ex_map (ndarray): n-dimensional array of the Ex component of the Jones vector.
            - Ey_map (ndarray): n-dimensional array of the Ey component of the Jones vector.
    """
    original_shape = s0_input.shape
    s0_flat = s0_input.flatten()
    s1_flat = s1_input.flatten()
    s2_flat = s2_input.flatten()
    s3_flat = s3_input.flatten()

    ex_flat = np.zeros_like(s0_flat, dtype = np.complex128)
    ey_flat = np.zeros_like(s0_flat, dtype = np.complex128)
    for ii, (s0, s1, s2, s3) in enumerate(zip(s0_flat, s1_flat, s2_flat, s3_flat)):
        s_vector = np.array([s0, s1, s2, s3])
        j_vector = pypolar.mueller.stokes_to_jones(s_vector)
        ex, ey = j_vector[0], j_vector[1]

        ex_flat[ii] = ex
        ey_flat[ii] = ey

    ex_map = ex_flat.reshape(original_shape)
    ey_map = ey_flat.reshape(original_shape)

    return ex_map, ey_map

def jones2stokes_pypol(ex_in, ey_in):
    """
    Convert Jones vector components (Ex and Ey) to Stokes parameters using PyPol.

    This function takes n-dimensional arrays of Jones vector components (Ex and Ey),
    flattens them, converts each pair of Ex and Ey to Stokes parameters (s0, s1, s2, s3)
    using the PyPol library, and reshapes the results back to the original input shape.

    Parameters:
        ex_in (ndarray): n-dimensional array of the Ex component of the Jones vector.
        ey_in (ndarray): n-dimensional array of the Ey component of the Jones vector.

    Returns:
        tuple: A tuple containing:
            - s0_map (ndarray): n-dimensional array of the S0 Stokes parameter.
            - s1_map (ndarray): n-dimensional array of the S1 Stokes parameter.
            - s2_map (ndarray): n-dimensional array of the S2 Stokes parameter.
            - s3_map (ndarray): n-dimensional array of the S3 Stokes parameter.
    """
    original_shape = ex_in.shape
    ex_flat = ex_in.flatten()
    ey_flat = ey_in.flatten()

    s0_flat = np.zeros_like(ex_flat, dtype = np.float64)
    s1_flat = np.zeros_like(ex_flat, dtype = np.float64)
    s2_flat = np.zeros_like(ex_flat, dtype = np.float64)
    s3_flat = np.zeros_like(ex_flat, dtype = np.float64)
    for ii, (ex, ey) in enumerate(zip(ex_flat, ey_flat)):
        j_vector = np.array([ex, ey])
        s_vector = pypolar.jones.jones_to_stokes(j_vector)
        s0, s1, s2, s3 = s_vector[0], s_vector[1], s_vector[2], s_vector[3]

        s0_flat[ii] = s0
        s1_flat[ii] = s1
        s2_flat[ii] = s2
        s3_flat[ii] = s3

    s0_map = s0_flat.reshape(original_shape)
    s1_map = s1_flat.reshape(original_shape)
    s2_map = s2_flat.reshape(original_shape)
    s3_map = s3_flat.reshape(original_shape)

    return s0_map, s1_map, s2_map, s3_map

def show_stokes_map(s0_map, s1_map, s2_map, s3_map, normalize=True):
    """
    Display Stokes parameter maps using Matplotlib.

    This function creates a 2x2 grid of subplots to visualize the Stokes parameters.
    The S0 parameter is displayed with a gray colormap, while the normalized S1, S2,
    and S3 parameters (divided by S0) are displayed with a bwr colormap.

    Parameters:
        s0_map (ndarray): n-dimensional array representing the S0 Stokes parameter.
        s1_map (ndarray): n-dimensional array representing the S1 Stokes parameter.
        s2_map (ndarray): n-dimensional array representing the S2 Stokes parameter.
        s3_map (ndarray): n-dimensional array representing the S3 Stokes parameter.

    Returns:
        fig (matplotlib.figure.Figure): The Matplotlib figure containing the subplots.
        axes (ndarray of Axes): Array of Matplotlib Axes objects corresponding to the subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)

    if normalize:
        titles = ['S0', 'S1/S0', 'S2/S0', 'S3/S0']
        maps = [s0_map, s1_map / s0_map, s2_map / s0_map, s3_map / s0_map]
    else:
        titles = ['S0', 'S1', 'S2', 'S3']
        maps = [s0_map, s1_map, s2_map, s3_map]
    # Loop through subplots and add images
    for i, (ax, data, title) in enumerate(zip(axes.flat, maps, titles)):
        if i == 0:  # For the first plot
            im = ax.imshow(data, cmap='gray')
            scalebar = ScaleBar(6, "µm", frameon=False, color='white', location="lower right", height_fraction=.02)
            ax.add_artist(scalebar)

        else:  # For other plots
            im = ax.imshow(data, cmap='bwr', vmin=-1, vmax=1)
            scalebar = ScaleBar(6, "µm", frameon=False, color='black', location="lower right", height_fraction=.02)
            ax.add_artist(scalebar)
        #ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation='vertical', shrink=1)

    plt.tight_layout()
    plt.show()
    return fig, axes

def map_angles_to_first_quadrant(angles):
    """
    Maps angles (in radians) to the first quadrant [0, π/2], preserving only
    their horizontality or verticality.

    Parameters:
        angles (array-like): Array of angles in radians.

    Returns:
        numpy.ndarray: Angles mapped to [0, π/2].
    """
    return np.arctan2(np.abs(np.sin(angles)), np.abs(np.cos(angles)))

def map_angles_to_first_and_fourth_quadrant(angles):
    """
    Maps angles (in radians) to the first and fourth quadrants [-π/2, π/2],
    preserving their orientation relative to the x-axis.

    Parameters:
        angles (array-like): Array of angles in radians.

    Returns:
        numpy.ndarray: Angles mapped to [-π/2, π/2].
    """
    return np.arctan2(np.sin(angles), np.abs(np.cos(angles)))



def ideal_pol_ori_az_ravel(XY, alpha_0, x0, y0):
    """
    Computes the ideal polarization orientation angles mapped into
    the first quadrant [0, π/2].
    For Azimuthal polarization the offset angle corresponds to 0.
    For Radial polarization the offset angle corresponds to pi/2.

    Parameters:
        XY (tuple): A tuple of arrays (X, Y) representing coordinates.
        alpha_0 (float): Offset angle (in radians) for the orientation.
        x0 (float): X-coordinate of the center.
        y0 (float): Y-coordinate of the center.

    Returns:
        numpy.ndarray: Flattened array of orientations mapped to [0, π/2].
    """
    X, Y = XY
    Xc = X - x0
    Yc = Y - y0

    orientation = np.arctan2(Xc, Yc) - alpha_0  # For Azimuthal
    orientation_first_quadrant = map_angles_to_first_quadrant(orientation)
    return orientation_first_quadrant.ravel()

def orientation_perpendicular_to_radius(XY, x0, y0, angle_deviation_rad):
    X, Y = XY
    Xc = X - x0
    Yc = Y - y0

    radial_angle = np.arctan2(Yc, Xc)
    azimuthal_angle = radial_angle + np.pi / 2 + angle_deviation_rad
    return map_angles_to_first_quadrant(azimuthal_angle)


def ideal_stokes_azimuthal(X, Y, x0, y0, rotation_angle_degrees):
    X = X - x0
    Y = Y - y0

    X, Y = rotate_coordinates(X, Y, rotation_angle_degrees)
    R, phi = np.sqrt(X**2 + Y**2), np.arctan2(X, Y)

    ex = + np.sin(phi)
    ey = + np.cos(phi)

    s0, s1, s2, s3 = jones2stokes_pypol(ex, ey)
    return s0, s1, s2, s3

def global_stokes_model(coords, x0, y0, rotation_angle):
    """
    Model function for global fitting. Computes the ideal Stokes parameters
    given the parameters to optimize (x0, y0, rotation_angle).
    """
    X, Y = coords
    S0, S1, S2, S3 = ideal_stokes_azimuthal(X, Y, x0, y0, rotation_angle)
    return np.concatenate([S1.flatten(), S2.flatten(), S3.flatten()])  # Flattened Stokes vectors


def rotate_coordinates(x, y, angle_degrees):
    """
    Rotate 2D points by a given angle around the origin applying the rotation matrix transformation.

    Parameters:
        x, y : array-like
            Arrays of x and y coordinates.
        angle_degrees : float
            Rotation angle in degrees (counterclockwise).

    Returns:
        tuple of numpy.ndarray
            Rotated x and y coordinates.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute cos and sin of the angle
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # Apply the rotation formula
    x_rotated = x * cos_theta - y * sin_theta
    y_rotated = x * sin_theta + y * cos_theta

    return x_rotated, y_rotated


def compute_azimuthal_angle_error_map(orientation_angles, s0, roi_search = 50, show_center_estimation = False):

    # Get the center of the beam and crop a ROI to exclude noise
    row_hole, col_hole = find_beam_hole_center(np.uint8(s0), roi_size = roi_search, show_estimation = show_center_estimation)
    semi_roi_size = 100
    orientation_angles = orientation_angles[row_hole - semi_roi_size : row_hole + semi_roi_size,
                                            col_hole - semi_roi_size : col_hole + semi_roi_size]

    # Define coordinate system centered on the beam
    x = np.arange(orientation_angles.shape[1], dtype = np.float64) - orientation_angles.shape[1] // 2
    y = np.arange(orientation_angles.shape[0], dtype = np.float64) - orientation_angles.shape[0] // 2
    X, Y = np.meshgrid(x, y)

    # Compute corresponding angles
    perfect_azimuthal_angle = np.arctan2(X, Y)

    # Compute errors and map them to the first and fourth quadrant [-pi/2, +pi/2]
    error_angle_map = orientation_angles - perfect_azimuthal_angle
    error_angle_map = map_angles_to_first_and_fourth_quadrant(error_angle_map)
    mean_absolute_error = np.mean(np.abs(error_angle_map))

    return error_angle_map, mean_absolute_error

def compute_ellipticity(list_vectors, vectors_type):
    """
    Computes the ellipticity of polarization states from input vectors.

    Parameters:
        list_vectors (list of np.ndarray): List containing two NumPy arrays (Ex and Ey components).
        vectors_type (str): The type of vectors provided. Currently supports 'Jones'.

    Returns:
        np.ndarray: The computed ellipticity values reshaped to the original input shape.
    """
    if vectors_type != 'Jones':
        raise ValueError("Unsupported vector type. Only 'Jones' is currently supported.")

    # Extract Ex and Ey components
    ex_arr, ey_arr = list_vectors
    if ex_arr.shape != ey_arr.shape:
        raise ValueError("Shape mismatch: Ex and Ey arrays must have the same dimensions.")

    original_shape = ex_arr.shape
    ex_arr_flat = ex_arr.flatten()
    ey_arr_flat = ey_arr.flatten()

    # Compute ellipticity for each pair of (Ex, Ey)
    ellipticity_values = np.array([pypolar.jones.ellipticity(np.array([ex, ey])) for ex, ey in zip(ex_arr_flat, ey_arr_flat)])

    # Reshape back to original shape
    return ellipticity_values.reshape(original_shape)