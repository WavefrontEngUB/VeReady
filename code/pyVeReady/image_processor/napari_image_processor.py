import matplotlib
matplotlib.use('TkAgg')  # instead of Qt
import napari
from napari.utils.colormaps import Colormap
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from magicgui import magicgui
from image_processing_functions import *
from hdf5_loader_widget import HDF5DropWidget
import numpy as np


# =====================================
# Image Processing Widget Definition
# =====================================
@magicgui(
    call_button="Process",
    excitation_layer={"label": "Excitation"},
    depletion_layer={"label": "Depletion"},
    widefield={"label": "Widefield"},
    confocal={"label": "Confocal"},
    ism={"label": "ISM Pixel Reassignment"},
    constant_subtraction={"label": "Constant subtraction"},
    adaptive_subtraction={"label": "Adaptive subtraction"},
    moving_integration_roi={"label": "Moving Integration ROI"},
    detector_pixel_size_nm={'label': 'Detector pixel size (nm)'},
    pinhole_fwhm_nm={"label": "Pinhole FWHM (µm)"},
    subtraction_factor={"label": "Subtraction factor", "min": 0.0, "max": 1.0, "step": 0.01},
    num_z_slices={"label": "Number of Z slices", "min": 1, "step": 1},
    x_step={"label": "X Scan Step (µm)", "min": 0.0, "step": 0.001},
    y_step={"label": "Y Scan Step (µm)", "min": 0.0, "step": 0.001},
    z_step={"label": "Z Scan Step (µm)", "min": 0.0, "step": 0.001},
)
def process_widget(
    viewer: napari.Viewer,
    excitation_layer: napari.layers.Image = None,
    depletion_layer: napari.layers.Image = None,
    widefield: bool = True,
    confocal: bool = False,
    ism: bool = False,
    constant_subtraction: bool = False,
    adaptive_subtraction: bool = False,
    moving_integration_roi: bool = False,
    detector_pixel_size_nm: float = 65,
    pinhole_fwhm_nm: float = 230,
    subtraction_factor: float = 0.6,
    num_z_slices: int = 1,
    x_step: float = 0.05,
    y_step: float = 0.05,
    z_step: float = 0.05,
):
    # Check if Data is selected
    if excitation_layer is None:
        return

    if (constant_subtraction or adaptive_subtraction) is True and depletion_layer is None:
        return

    # Convert available data to float datatype
    datatype = np.float32
    if excitation_layer is not None:
        excitation_data = np.array(excitation_layer.data, dtype=datatype)
        excitation_data, _ = subtract_background(excitation_data)
        excitation_data = normalize_stack(excitation_data)
        if moving_integration_roi:
            excitation_data, _ = extract_centered_rois(excitation_data, 15)

    if depletion_layer is not None:
        depletion_data = np.array(depletion_layer.data, dtype=datatype)
        depletion_data, _ = subtract_background(depletion_data)
        depletion_data = normalize_stack(depletion_data)
        if moving_integration_roi:
            depletion_data, _ = extract_centered_rois(depletion_data, 15)

    # Compute desired Image Processing Algorithms
    if widefield:
        widefield_image = process_z_stack(
            excitation_data,
            num_z_slices=num_z_slices,
            processing_func=compute_integrated_image,
        )
        viewer.add_image(
            widefield_image,
            name='Widefield Reconstruction',
            scale=(z_step, y_step, x_step)
        )

    if confocal:
        confocal_image = process_z_stack(
            excitation_data,
            num_z_slices=num_z_slices,
            processing_func=compute_confocal_integrated_image,
            pinhole_fwhm=pinhole_fwhm_nm,
            detector_pixel_size=detector_pixel_size_nm,
        )
        viewer.add_image(
            confocal_image,
            name='Confocal Reconstruction',
            scale=(z_step, y_step, x_step)
        )

    if constant_subtraction:
        subtraction_confocal_image = process_z_stack(
            excitation_data,
            num_z_slices=num_z_slices,
            processing_func=compute_subtraction_confocal_image,
            stack_depletion=depletion_data,
            alpha=subtraction_factor,
            use_adaptative_alpha=False,
            pinhole_fwhm=pinhole_fwhm_nm,
            detector_pixel_size=detector_pixel_size_nm,
        )
        viewer.add_image(
            subtraction_confocal_image,
            name='Subtraction Confocal Reconstruction',
            scale=(z_step, y_step, x_step)
        )

    if adaptive_subtraction:
        adaptive_subtraction_confocal_image = process_z_stack(
            excitation_data,
            num_z_slices=num_z_slices,
            processing_func=compute_subtraction_confocal_image,
            stack_depletion=depletion_data,
            alpha=subtraction_factor,
            use_adaptative_alpha=True,
            pinhole_fwhm=pinhole_fwhm_nm,
            detector_pixel_size=detector_pixel_size_nm,
        )
        viewer.add_image(
            adaptive_subtraction_confocal_image,
            name='Adaptive Subtraction Confocal Reconstruction',
            scale=(z_step, y_step, x_step)
        )


# =====================================
# New Widget: Voxel Size Adjustment
# =====================================
@magicgui(
    call_button="Apply scale",
    layer={"label": "Select image layer"},
    z_step={"label": "Z step (µm)", "min": 0.0, "step": 0.001},
    y_step={"label": "Y step (µm)", "min": 0.0, "step": 0.001},
    x_step={"label": "X step (µm)", "min": 0.0, "step": 0.001},
)
def voxel_size_widget(
    viewer: napari.Viewer,
    layer: napari.layers.Image = None,
    z_step: float = 1.0,
    y_step: float = 0.065,
    x_step: float = 0.065,
):
    """Widget to adjust voxel (scale) size interactively."""
    if layer is None:
        return
    layer.scale = (z_step, y_step, x_step)
    print(f"Updated voxel size for {layer.name}: (z={z_step}, y={y_step}, x={x_step})")


# =====================================
# Napari Execution
# =====================================
if __name__ == "__main__":
    viewer = napari.Viewer()

    # Add Custom Widgets
    viewer.window.add_dock_widget(process_widget, area="right", name='Image Processor')
    viewer.window.add_dock_widget(voxel_size_widget, area="right", name="Voxel Size Adjuster")

    hdf5_widget = HDF5DropWidget(viewer)
    viewer.window.add_dock_widget(hdf5_widget, area="right", name="HDF5 Loader")

    # Load Custom Colormaps
    red_hot_colormap = Colormap(read_imagej_lut('Red Hot.lut'), name='Red Hot')
    AVAILABLE_COLORMAPS["Red Hot"] = red_hot_colormap

    napari.run()
