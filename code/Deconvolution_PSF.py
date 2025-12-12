import numpy as np
import tifffile as tiff
from scipy.ndimage import zoom, gaussian_filter, map_coordinates
from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy, wiener, unsupervised_wiener
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cmasher as cmr

# -------------------------------
# USER CONFIG
# -------------------------------
tif_path = r"C:/Users/laboratori/OneDrive - Universitat de Barcelona (1)/Data/20251201-Artur__23nm_6pxl/20251201-Artur_10nm_6pxls_bkg_subtracted.tif"
measured_pixel_size_um = 0.01  # 10 nm per pixel
TARGET_PIXEL_SIZE = 0.01       # µm
n_iter_rl = 10
bead_sigma_nm = 23             # Gaussian bead sigma
smooth_sigma_um = 0            # Gaussian smoothing sigma in µm (0 to skip)

# -------------------------------
# LOAD MEASURED PSF
# -------------------------------
psf_meas = tiff.imread(tif_path).astype(float)
if psf_meas.ndim == 3 and psf_meas.shape[0] == 1:
    psf_meas = psf_meas[0]

if smooth_sigma_um > 0:
    psf_meas = gaussian_filter(psf_meas, sigma=smooth_sigma_um/measured_pixel_size_um)

psf_meas /= psf_meas.max()
print("Loaded PSF shape:", psf_meas.shape)

# -------------------------------
# CREATE GAUSSIAN BEAD KERNEL
# -------------------------------
px_size_nm = measured_pixel_size_um * 1000
sigma_px = (bead_sigma_nm / 2) / px_size_nm
r_px = int(np.ceil(3*sigma_px))
x = np.arange(-r_px, r_px + 1)
xx, yy = np.meshgrid(x, x)
bead_kernel = np.exp(-(xx**2 + yy**2)/(2*sigma_px**2))
bead_kernel /= bead_kernel.sum()
print(f"Gaussian bead kernel created with sigma {sigma_px:.2f} px")

# -------------------------------
# SELECT PSF CENTER
# -------------------------------
plt.figure(figsize=(6,6))
plt.title("Measured PSF (CLICK CENTER)")
plt.imshow(psf_meas, cmap='inferno', interpolation='nearest')
plt.colorbar(label="Normalized intensity")
coords = plt.ginput(1, timeout=0)
plt.close('all')
if len(coords) == 0:
    raise RuntimeError("No click received. Run again and click on the measured PSF.")
cx_click, cy_click = coords[0]
center = (int(round(cy_click)), int(round(cx_click)))
print("Selected center (row, col):", center)

# -------------------------------
# RESAMPLING & RADIAL AVERAGE
# -------------------------------
def resample_to_target(data, current_psize_um, target_psize_um):
    if np.isclose(current_psize_um, target_psize_um):
        return data
    scale = current_psize_um / target_psize_um
    return zoom(data, zoom=scale, order=3)

def radial_average(data, center, pixel_size_um, Rmax_um=0.4, step_deg=1):
    cy, cx = center
    ny, nx = data.shape
    Rmax = int(Rmax_um / pixel_size_um)
    r_pix = np.linspace(-Rmax, Rmax, 2*Rmax + 1)
    angles = np.arange(0, 360, step_deg)
    D = np.zeros((len(r_pix), len(angles)))
    for i, ang in enumerate(angles):
        theta = np.deg2rad(ang)
        xs = cx + r_pix*np.cos(theta)
        ys = cy + r_pix*np.sin(theta)
        coords = np.vstack((ys, xs))
        D[:, i] = map_coordinates(data, coords, order=1, mode='constant', cval=0)
    center_idx = len(r_pix)//2
    D = np.delete(D, center_idx, axis=0)
    r_pix = np.delete(r_pix, center_idx)
    profile = np.mean(D, axis=1)
    return r_pix * pixel_size_um, profile

# -------------------------------
# RESAMPLE
# -------------------------------
psf_meas_r = resample_to_target(psf_meas, measured_pixel_size_um, TARGET_PIXEL_SIZE)
bead_kernel_r = resample_to_target(bead_kernel, measured_pixel_size_um, TARGET_PIXEL_SIZE)
scale = measured_pixel_size_um / TARGET_PIXEL_SIZE
center_resampled = (int(round(center[0]*scale)), int(round(center[1]*scale)))

# -------------------------------
# DEFINE DECONVOLUTION METHODS
# -------------------------------
def do_rl(psf):
    out = richardson_lucy(psf, bead_kernel, num_iter=n_iter_rl, clip=True)
    return out

def do_wiener(psf):
    out = wiener(psf, bead_kernel, balance=0.01)
    return np.clip(out, 0, None)

def do_unsupervised(psf):
    out, _ = unsupervised_wiener(psf, bead_kernel)
    return np.clip(out, 0, None)

methods = {
    "RL": do_rl,
    "Wiener": do_wiener,
    "Unsupervised Wiener": do_unsupervised,
}

results = {}

# -------------------------------
# PROCESS EACH METHOD
# -------------------------------
for name, func in methods.items():
    print("Running", name)
    deconv = func(psf_meas)
    deconv /= deconv.max()
    reconv = convolve2d(deconv, bead_kernel, mode='same')
    reconv /= reconv.max()

    # Resample
    deconv_r = resample_to_target(deconv, measured_pixel_size_um, TARGET_PIXEL_SIZE)
    reconv_r = resample_to_target(reconv, measured_pixel_size_um, TARGET_PIXEL_SIZE)

    # Radial profiles
    RADIUS_TO_show_um = 0.4
    r_meas, profile_meas = radial_average(psf_meas_r, center_resampled, TARGET_PIXEL_SIZE, Rmax_um=RADIUS_TO_show_um)
    r_deconv, profile_deconv = radial_average(deconv_r, center_resampled, TARGET_PIXEL_SIZE, Rmax_um=RADIUS_TO_show_um)
    r_reconv, profile_reconv = radial_average(reconv_r, center_resampled, TARGET_PIXEL_SIZE, Rmax_um=RADIUS_TO_show_um)

    profile_meas /= profile_meas.max()
    profile_deconv /= profile_deconv.max()
    profile_reconv /= profile_reconv.max()

    results[name] = {
        "deconv": deconv_r,
        "reconv": reconv_r,
        "r": r_meas,
        "meas_prof": profile_meas,
        "dec_prof": profile_deconv,
        "rec_prof": profile_reconv,
    }

# -------------------------------
# PLOTTING
# -------------------------------
um_to_nm = 1000
extent_psf_um = [0, psf_meas_r.shape[1]*TARGET_PIXEL_SIZE, 0, psf_meas_r.shape[0]*TARGET_PIXEL_SIZE]
extent_kernel_um = [-(bead_kernel_r.shape[1]-1)/2*TARGET_PIXEL_SIZE, (bead_kernel_r.shape[1]-1)/2*TARGET_PIXEL_SIZE,
                    -(bead_kernel_r.shape[0]-1)/2*TARGET_PIXEL_SIZE, (bead_kernel_r.shape[0]-1)/2*TARGET_PIXEL_SIZE]

n_methods = len(methods)
fig, axes = plt.subplots(2, n_methods+1, figsize=(6*(n_methods+1), 10))

# Measured PSF
a0 = axes[0,0]
a0.imshow(psf_meas_r, cmap=cmr.sunburst, interpolation='nearest', extent=extent_psf_um, vmin=0, vmax=1)
a0.set_title("Measured PSF")
plt.colorbar(a0.images[0], ax=a0)

# Gaussian bead kernel
a1 = axes[1,0]
a1.imshow(bead_kernel_r, cmap=cmr.sunburst, interpolation='nearest', extent=extent_kernel_um)
a1.set_title("Bead kernel")
plt.colorbar(a1.images[0], ax=a1)

# Fill deconvolutions and reconvolutions
col = 1
for name in methods:
    deconv = results[name]["deconv"]
    reconv = results[name]["reconv"]

    ax_d = axes[0, col]
    ax_d.imshow(deconv, cmap=cmr.sunburst, extent=extent_psf_um, vmin=0, vmax=1)
    ax_d.set_title(f"{name} deconvolution")
    plt.colorbar(ax_d.images[0], ax=ax_d)

    ax_r = axes[1, col]
    ax_r.imshow(reconv, cmap=cmr.sunburst, extent=extent_psf_um, vmin=0, vmax=1)
    ax_r.set_title(f"{name} reconvolution")
    plt.colorbar(ax_r.images[0], ax=ax_r)

    col += 1

plt.tight_layout()
plt.show()

# -------------------------------
# RADIAL PROFILE PLOTS
# -------------------------------
for name in methods:
    r = results[name]["r"]
    plt.figure(figsize=(7,5))
    plt.plot(r*um_to_nm, results[name]["meas_prof"], label="Measured")
    plt.plot(r*um_to_nm, results[name]["dec_prof"], label=f"{name} Deconv")
    plt.plot(r*um_to_nm, results[name]["rec_prof"], label=f"{name} Reconvolved")
    plt.xlabel("Distance (nm)")
    plt.ylabel("Normalized intensity")
    plt.title(f"Radial Profiles - {name}")
    plt.legend()
    plt.show()
