import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
from skimage import data

def apply_gaussian_blur(image, sigma=2):
    """Apply Gaussian blur to a grayscale image."""
    blurred_image = gaussian_filter(image, sigma=sigma)  # Apply Gaussian blur
    return blurred_image

# Load sample grayscale image from skimage
test_image = plt.imread('pepe_boxeo_bin.png')[:,:,1] < 0.5
test_image = 255 * test_image / test_image.max()
test_image = np.uint8(test_image)
sigma = 7  # Standard deviation for Gaussian kernel
blurred = apply_gaussian_blur(test_image, sigma)

# Save images
imageio.imwrite("original_image.png", test_image)
imageio.imwrite("blurred_image.png", blurred)

# Display images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(test_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(blurred, cmap='gray')
axes[1].set_title("Blurred Image")
axes[1].axis("off")

plt.show()