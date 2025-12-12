import numpy as np
import matplotlib.pyplot as plt


def read_imagej_lut(path):
    """Read ImageJ .lut file into Nx3 array of floats (0–1)."""
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 3
    lut = np.frombuffer(data, dtype=np.uint8).reshape((3, n)).T
    return lut / 255.0


def subtract_background(stack, percentile=5):
    """
    Subtract a scalar background estimated from the entire 3D image stack.

    Parameters:
        stack (ndarray): 3D image stack of shape (N, Y, X).
        percentile (float): Percentile value (e.g., 5) to estimate scalar background level.

    Returns:
        stack_subtracted (ndarray): Stack after scalar background subtraction.
        background_scalar (float): Scalar background value subtracted from the stack.
    """
    background_scalar = np.percentile(stack, percentile)
    stack_subtracted = stack - background_scalar
    stack_subtracted = np.clip(stack_subtracted, 0, None)  # Remove negative values
    return stack_subtracted, background_scalar


def normalize_stack(stack, percentile=99, clip=False):
    """
    Normalizes a 3D image stack based on a chosen high percentile value.

    Parameters:
        stack (ndarray): 3D image stack (N, Y, X).
        percentile (float): Percentile value to normalize by (e.g., 99.9).
        clip (bool): If True, clip values exceeding normalization value to 1.

    Returns:
        normalized_stack (ndarray): Normalized image stack.
    """
    norm_val = np.percentile(stack, percentile)
    if norm_val == 0:
        raise ValueError("Normalization value is zero. Check input stack or choose different percentile.")
    normalized = stack / norm_val
    if clip:
        normalized = np.clip(normalized, 0, 1)
    return normalized


def compute_integrated_image(stack, desnake=False, substitute_first_row=False):
    """
    Computes a square image by summing pixel intensities along the first axis 0.

    Parameters:
        stack (ndarray): 3D image stack of shape (N, Y, X). N has to be a perfect square, meaning a square scan.
        desnake (bool): Whether to reverse every other row of the final image.
        substitute_first_row (bool): Whether to set the first column to the minimum value.

    Returns:
        image (ndarray): 2D image of shape (sqrt(N), sqrt(N))
    """
    pixel_values = stack.sum(axis=(1, 2))
    side = int(np.sqrt(stack.shape[0]))
    image = pixel_values.reshape((side, side))

    if substitute_first_row:
        image[:, 0] = image.min()

    if desnake:
        image[1::2] = np.flip(image[1::2], axis=1)

    return image


def generate_pinhole(stack, pinhole_fwhm, detector_pixel_size, show_plot=False):
    """
    Computes a Gaussian pinhole mask from a 3D image stack.

    Parameters:
        stack (ndarray): 3D image stack of shape (N, Y, X).
        pinhole_fwhm (float): Full width at half maximum of the Gaussian mask.
        detector_pixel_size (float): Detector pixel size.
        show_plot (bool): If True, displays the fingerprint and the pinhole mask.

    Returns:
        pinhole (ndarray): 2D Gaussian pinhole mask.
        fingerprint (ndarray): 2D summed projection of the stack.
        x0, y0 (int): Coordinates of the estimated center.
    """
    fingerprint = np.sum(stack, axis=0)
    y0, x0 = np.unravel_index(np.argmax(fingerprint), fingerprint.shape)

    rows, cols = fingerprint.shape
    y, x = np.ogrid[:rows, :cols]
    sigma = (pinhole_fwhm / 2.3548) / detector_pixel_size
    pinhole = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    if show_plot:
        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
        axs[0].imshow(fingerprint, cmap='hot')
        axs[0].scatter(x0, y0, color='red', marker='x', label='Estimated Center')
        axs[0].set_title('Fingerprint')
        axs[0].legend()
        axs[1].imshow(pinhole, cmap='hot')
        axs[1].set_title('Gaussian Pinhole')
        plt.tight_layout()
        plt.show()

    return pinhole, fingerprint, x0, y0


def compute_confocal_integrated_image(stack, pinhole_fwhm, detector_pixel_size, desnake=False, substitute_first_row=False):
    """
    Computes a square image by summing pixel intensities multiplied by a gaussian mask that acts as a confocal pinhole,
    along the first axis 0.

    Parameters:
        stack (ndarray): 3D image stack of shape (N, Y, X). N has to be a perfect square, meaning a square scan.
        desnake (bool): Whether to reverse every other row of the final image.
        substitute_first_row (bool): Whether to set the first column to the minimum value.

    Returns:
        image (ndarray): 2D image of shape (sqrt(N), sqrt(N))
    """
    pinhole, fingerprint, x0, y0 = generate_pinhole(stack, pinhole_fwhm, detector_pixel_size, show_plot=False)
    stack_confocal = stack * pinhole
    confocal_image = compute_integrated_image(stack_confocal, desnake, substitute_first_row)

    return confocal_image

def process_z_stack(stack, num_z_slices=1, processing_func=None, **kwargs):
    """
    Processes a 3D stack slice by slice using a given processing function.

    Parameters:
        stack (ndarray): 3D image stack of shape (N, Y, X)
        num_z_slices (int): Number of Z slices
        processing_func (callable): Function to apply to each Z slice
        **kwargs: Additional keyword arguments passed to processing_func

    Returns:
        z_images (ndarray): Array of processed images for each Z slice
    """
    if processing_func is None:
        raise ValueError("You must provide a processing_func")

    total_scan_positions = stack.shape[0]
    xy_size = int(np.sqrt(total_scan_positions // num_z_slices))
    assert xy_size * xy_size * num_z_slices == total_scan_positions, "Scan dimensions don't match (check Z slices input)."

    slice_length = xy_size * xy_size
    z_images = []

    for z in range(num_z_slices):
        data_z_slice = stack[z * slice_length:(z + 1) * slice_length, :, :]
        z_slice_image = processing_func(data_z_slice, **kwargs)
        z_images.append(z_slice_image)

    return np.array(z_images)


def compute_subtraction_confocal_image(stack_excitation, stack_depletion, alpha, use_adaptative_alpha, pinhole_fwhm, detector_pixel_size, desnake=False, substitute_first_row=False):
    if use_adaptative_alpha:
        im_excitation = compute_integrated_image(stack_excitation)
        im_excitation, _ = subtract_background(im_excitation)
        im_excitation = normalize_stack(im_excitation)

        im_depletion = compute_integrated_image(stack_depletion)
        im_depletion, _ = subtract_background(im_depletion)
        im_depletion = normalize_stack(im_depletion)

        alpha = 0.5 * ((im_excitation - im_depletion) + 1)
        alpha = alpha.flatten()[:, None, None]
    stack_subtraction = stack_excitation - alpha * stack_depletion
    stack_subtraction = np.clip(stack_subtraction, 0, None)
    pinhole, fingerprint, x0, y0 = generate_pinhole(stack_subtraction, pinhole_fwhm, detector_pixel_size, show_plot=False)
    stack_subtraction_confocal = stack_subtraction * pinhole
    subtraction_confocal_image = compute_integrated_image(stack_subtraction_confocal, desnake, substitute_first_row)

    return subtraction_confocal_image


def extract_centered_rois(stack: np.ndarray, roi_size: int) -> np.ndarray:
    """
    Extract square ROIs centered on the maximum pixel of each slice in a 3D stack.

    Parameters:
        stack (ndarray): 3D image stack of shape (N, Y, X).
        roi_size (int): Size of the square ROI (recommended odd number).

    Returns:
        rois (ndarray): Array of shape (N, roi_size, roi_size) containing cropped ROIs.
        positions (list of tuple): List of (row, col) coordinates of the maxima for each slice.
    """
    N, rows, cols = stack.shape
    half = roi_size // 2
    rois = []
    positions = []

    for i in range(N):
        slice_ = stack[i]
        # find max position
        max_pos = np.unravel_index(np.argmax(slice_), slice_.shape)
        r, c = max_pos
        positions.append((r, c))

        # compute ROI bounds
        r_start = max(r - half, 0)
        r_end   = min(r + half + 1, rows)
        c_start = max(c - half, 0)
        c_end   = min(c + half + 1, cols)

        # extract with padding if needed
        roi = np.zeros((roi_size, roi_size), dtype=stack.dtype)
        rr_start = half - (r - r_start)
        rr_end   = rr_start + (r_end - r_start)
        cc_start = half - (c - c_start)
        cc_end   = cc_start + (c_end - c_start)

        roi[rr_start:rr_end, cc_start:cc_end] = slice_[r_start:r_end, c_start:c_end]
        rois.append(roi)

    return np.array(rois), positions