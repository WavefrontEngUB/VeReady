import numpy as np
import matplotlib.pyplot as plt

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