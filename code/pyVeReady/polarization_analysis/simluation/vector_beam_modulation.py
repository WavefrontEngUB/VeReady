from pyVeReady.polarization_analysis.polarization_analysis_functions import *
from scipy.ndimage import rotate
from scipy.optimize import curve_fit


limits = 1
n_points_grid = 200
number_arrows = 30

x = np.linspace(-limits, limits, n_points_grid)
y = np.linspace(-limits, limits, n_points_grid)
X, Y = np.meshgrid(x, y)
R, phi = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)

input_polarization_angle_degrees = 45
ex_in = np.cos(input_polarization_angle_degrees * np.pi / 180) * np.ones_like(X, dtype = np.complex128)
ey_in = np.sin(input_polarization_angle_degrees * np.pi / 180) * np.ones_like(X, dtype = np.complex128)

fig, axes, _, _, _, _, _ = polarization_map_analysis([ex_in, ey_in], vector_tpye ='Jones',
                                                     show_relative_phase_map = True,
                                                     show_azimuthal_orientation_map= True,
                                                     draw_step = n_points_grid // number_arrows)
image_input = axes[0].images[0]
image_input.set_clim(vmin = 0, vmax = 1)
fig.suptitle('Input Beam')
fig.tight_layout()

x_modulation = np.sin(phi)
y_modulation = np.cos(phi)

modulation_rotation_angle = 15
x_modulation = rotate(x_modulation, modulation_rotation_angle, reshape = False, mode = 'nearest')
y_modulation = rotate(y_modulation, modulation_rotation_angle, reshape = False, mode = 'nearest')

fig_mod, axes_mod = plt.subplots(2, 2, sharex = True, sharey = True)
data_show = [np.abs(x_modulation), np.abs(y_modulation), np.angle(x_modulation), np.angle(y_modulation)]
titles_show = ['Abs X', 'Abs Y', 'Angle X', 'Angle Y']
for ii, ax in enumerate(axes_mod.flatten()):
    ax.imshow(data_show[ii], cmap = 'gray')
    ax.set_title(titles_show[ii])

ex_out = ex_in * x_modulation
ey_out = ey_in * y_modulation

fig_out, axes_out, _, _, relative_phase, orientation_map_first_quadrant, _ = polarization_map_analysis([ex_out, ey_out], vector_tpye ='Jones',
                                                                                                       show_relative_phase_map = True,
                                                                                                       show_azimuthal_orientation_map= True,
                                                                                                       orientation_into_first_quadrant = True,
                                                                                                       draw_step = n_points_grid // number_arrows)
image_output = axes_out[0].images[0]
image_output.set_clim(vmin = 0, vmax = 1)
fig_out.suptitle('Output Beam')
fig_out.tight_layout()

# Perform Fitting to Perfect Azimuthal, angles must be mapped into the first quadrant
x = np.arange(ex_out.shape[1], dtype = np.float64) - ex_out.shape[1] // 2
y = np.arange(ex_out.shape[0], dtype = np.float64) - ex_out.shape[0] // 2
X, Y = np.meshgrid(x, y)

[popt, pcov] = curve_fit(ideal_pol_ori_az_ravel,
                        (X.ravel(), Y.ravel()),
                        orientation_map_first_quadrant.ravel(),
                        (0, 0, 0))

print(f'Angle: {popt[0]*180/np.pi:.3f} Degrees')
