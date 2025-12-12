import numpy as np
from scipy.signal import fftconvolve
from tifffile import imwrite
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for plotting
import matplotlib.pyplot as plt

# -------------------------
# USER PARAMETERS
# -------------------------
DATA_FILE = "C:/Users/laboratori/Desktop/PySLM_VeReady/pyVeReady/misc/Artur2.0/nterm8.npz"  # your npz
LAMBDA_UM = 0.488        # wavelength in um
pixel_nm = 10            # grid spacing in nm
pixel_um = pixel_nm * 1e-3
PATCH_PX = 50            # patch size in pixels
bead_sigma_nm = 170/2       # FWHM of bead in nm (user can change)
out_dir = "scan_output4"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Derived bead parameters
# -------------------------
bead_sigma_um = bead_sigma_nm * 1e-3  # convert to microns

# -------------------------
# Load NPZ radial data
# -------------------------
data = np.load(DATA_FILE, allow_pickle=True)
irz2 = data["irz2"]        # radial x z intensity map
iaz2 = data["iaz2"]
NPZ = int(data["NPZ"])
zonaVisPp2P = 6 #float(data["zonaVisPp2P"])  # in units of lambda

# -------------------------
# Build radial profile at z=0
# -------------------------
z_index = NPZ // 2
r_slice = irz2.shape[0] // 2
I_rz0 = irz2[r_slice:, z_index].astype(float)
I_rz0 /= I_rz0.max()

r_vals_lambda = np.linspace(0, zonaVisPp2P, len(I_rz0))
r_vals_um = r_vals_lambda * LAMBDA_UM

# -------------------------
# Build XY grid covering the patch
# -------------------------
patch_half_um = (PATCH_PX // 2) * pixel_um
margin_um = 1.0 * pixel_um * PATCH_PX
FOV_patch_um = 2 * (patch_half_um + margin_um)
N_patch = int(np.round(FOV_patch_um / pixel_um))
if N_patch % 2 == 0:
    N_patch += 1

coords = (np.arange(N_patch) - (N_patch // 2)) * pixel_um
X, Y = np.meshgrid(coords, coords)
R = np.sqrt(X**2 + Y**2)

# Interpolate radial profile onto XY grid
I_xy_full = np.interp(R.ravel(), r_vals_um, I_rz0, left=0.0, right=0.0).reshape(R.shape)
I_xy_full /= I_xy_full.max()

# -------------------------
# Extract central patch
# -------------------------
center_idx = N_patch // 2
x_start = center_idx - (PATCH_PX // 2)
x_end = x_start + PATCH_PX
y_start = x_start
y_end = y_start + PATCH_PX
beam_patch = I_xy_full[y_start:y_end, x_start:x_end].copy()
assert beam_patch.shape == (PATCH_PX, PATCH_PX), f"Patch shape mismatch: {beam_patch.shape}"

# -------------------------
# Build Gaussian bead kernel
# -------------------------
sigma_px = bead_sigma_um / pixel_um
kernel_half = int(np.ceil(sigma_px))
kernel_size =  kernel_half
kx = (np.arange(kernel_size) - kernel_half) * pixel_um
KX, KY = np.meshgrid(kx, kx)
KR2 = KX**2 + KY**2
bead_kernel = np.exp(-0.5 * KR2 / (bead_sigma_um**2))
bead_kernel /= bead_kernel.sum()
bead_kernel = KR2 <= (sigma_px**2)
# -------------------------
# Convolution: beam_patch * bead_kernel
# -------------------------
psf_conv = fftconvolve(beam_patch, bead_kernel, mode='same')

# -------------------------
# Simulate scanning (line-by-line)
# -------------------------
scan_positions = []
scan_values = []
scan_stack = np.zeros((PATCH_PX * PATCH_PX, PATCH_PX, PATCH_PX), dtype=np.float32)
frame_idx = 0
for iy in range(PATCH_PX):
    for ix in range(PATCH_PX):
        scan_positions.append((ix, iy))
        scan_values.append(psf_conv[iy, ix])
        # build shifted bead mask for this frame
        kernel_full = np.zeros_like(beam_patch)
        ky0 = iy - kernel_half
        kx0 = ix - kernel_half
        k_y1 = max(0, -ky0)
        k_y2 = min(kernel_size, PATCH_PX - ky0)
        k_x1 = max(0, -kx0)
        k_x2 = min(kernel_size, PATCH_PX - kx0)
        p_y1 = max(0, ky0)
        p_y2 = p_y1 + (k_y2 - k_y1)
        p_x1 = max(0, kx0)
        p_x2 = p_x1 + (k_x2 - k_x1)
        if (k_y2 - k_y1) > 0 and (k_x2 - k_x1) > 0:
            kernel_full[p_y1:p_y2, p_x1:p_x2] = bead_kernel[k_y1:k_y2, k_x1:k_x2]
        scan_stack[frame_idx] = (beam_patch * kernel_full).astype(np.float32)
        frame_idx += 1

scan_values = np.array(scan_values)
scan_positions = np.array(scan_positions)

# -------------------------
# Save outputs
# -------------------------
imwrite(os.path.join(out_dir, "beam_patch.tif"), (beam_patch / beam_patch.max()).astype(np.float32))
imwrite(os.path.join(out_dir, "psf_convolved.tif"), (psf_conv / psf_conv.max()).astype(np.float32))
imwrite(os.path.join(out_dir, "scan_stack.tif"), (scan_stack / scan_stack.max()).astype(np.float32))
np.savetxt(os.path.join(out_dir, "scan_values.txt"), scan_values)
np.savetxt(os.path.join(out_dir, "scan_positions.txt"), scan_positions, fmt="%d")

print("Saved outputs in folder:", out_dir)
print("beam_patch shape:", beam_patch.shape)
print("psf_conv shape:", psf_conv.shape)
print("scan frames:", scan_stack.shape[0])
print("Example: center PSF value:", psf_conv[PATCH_PX//2, PATCH_PX//2])

# -------------------------
# Visualization: 3-panel subplot
# -------------------------
axis_nm = (np.arange(PATCH_PX) - PATCH_PX//2) * pixel_nm
center_idx = PATCH_PX // 2

fig, axs = plt.subplots(1, 3, figsize=(18,5))

# 1) Beam patch
im0 = axs[0].imshow(beam_patch, extent=[axis_nm[0], axis_nm[-1], axis_nm[0], axis_nm[-1]],
                    origin='lower', cmap='inferno')
axs[0].axhline(0, color='white', linestyle='--', lw=1)
axs[0].axvline(0, color='white', linestyle='--', lw=1)
axs[0].set_xlabel("X (nm)")
axs[0].set_ylabel("Y (nm)")
axs[0].set_title("Beam Patch")
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="Normalized intensity")

# 2) Convolved PSF
im1 = axs[1].imshow(psf_conv, extent=[axis_nm[0], axis_nm[-1], axis_nm[0], axis_nm[-1]],
                    origin='lower', cmap='inferno')
axs[1].axhline(0, color='white', linestyle='--', lw=1)
axs[1].axvline(0, color='white', linestyle='--', lw=1)
axs[1].set_xlabel("X (nm)")
axs[1].set_ylabel("Y (nm)")
axs[1].set_title("Convolved PSF")
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Normalized intensity")

# 3) Line profile through center
axs[2].plot(axis_nm, beam_patch[center_idx, :], 'b-', lw=2, label='Beam Patch')
axs[2].plot(axis_nm, psf_conv[center_idx, :], 'r-', lw=2, label='Convolved PSF')
axs[2].set_xlabel("X (nm) through center")
axs[2].set_ylabel("Normalized intensity")
axs[2].set_title("Central Line Profile")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
