import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

def centered_FFT2(field):
    """Compute the centered FFT of a 2D field."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

# Define grid
N = 5000  # Increased grid size for better sampling at focal plane
L = 2   # Field of view
x_range = np.linspace(-L/2, L/2, N)
y_range = np.linspace(-L/2, L/2, N)
x, y = np.meshgrid(x_range, y_range)

# Define circular aperture
w0 = 0.025  # Beam waist
r = np.sqrt(x**2 + y**2)
aperture = np.where(r <= w0, 1, 0)

# Uniform phase beam (Gaussian beam precursor)
uniform_phase = aperture
E_gaussian = centered_FFT2(uniform_phase)
intensity_gaussian = np.abs(E_gaussian)**2

# Vortex phase beam (LG_0^1 precursor)
vortex_phase = aperture * np.exp(1j * np.angle(x + 1j*y))
E_lg01 = centered_FFT2(vortex_phase)
intensity_lg01 = np.abs(E_lg01)**2

# Define a Cyan Hot colormap (Black -> Blue -> White)
cyan_hot_colors = [(0, 0, 0), (0, 0, 1), (1, 1, 1)]
cyan_hot_cmap = mcolors.LinearSegmentedColormap.from_list("cyan_hot", cyan_hot_colors, N=256)

# Define a Red Hot colormap (Black -> Red -> White)
red_hot_colors = [(0, 0, 0), (1, 0, 0), (1, 1, 1)]
red_hot_cmap = mcolors.LinearSegmentedColormap.from_list("red_hot", red_hot_colors, N=256)

# Define zoom region
zoom_L = 0.08  # Total zoomed field of view (x and y range from -0.04 to 0.04)
zoom_N = int(N * (zoom_L / L))
center = N // 2
zoom_slice = slice(center - zoom_N // 2, center + zoom_N // 2)

# Crop images
zoomed_intensity_gaussian = intensity_gaussian[zoom_slice, zoom_slice]
zoomed_intensity_lg01 = intensity_lg01[zoom_slice, zoom_slice]

# Save images directly (without figure canvas)
imageio.imwrite("gaussian_beam_zoomed.png", (zoomed_intensity_gaussian / zoomed_intensity_gaussian.max() * 255).astype(np.uint8))
imageio.imwrite("lg01_beam_zoomed.png", (zoomed_intensity_lg01 / zoomed_intensity_lg01.max() * 255).astype(np.uint8))

# Plot and display zoomed images
plt.figure(figsize=(6, 5))
plt.imshow(zoomed_intensity_gaussian, origin='lower', cmap=cyan_hot_cmap)
plt.colorbar(label='Intensity')
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian Beam Intensity (Zoomed)")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(zoomed_intensity_lg01, origin='lower', cmap=red_hot_cmap)
plt.colorbar(label='Intensity')
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Laguerre-Gaussian 01 Beam Intensity (Zoomed)")
plt.show()