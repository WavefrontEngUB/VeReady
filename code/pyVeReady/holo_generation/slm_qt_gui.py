from magicgui import magicgui
from qtpy.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout, QWidget,
    QCheckBox, QSpinBox, QLabel
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import qdarkstyle
import matplotlib.pyplot as plt
from hologram_computation_V2 import *
import slmpy


def main():
    # Initialize Qt Application
    app = QApplication([])

    # Apply dark theme to Qt
    app.setStyleSheet(qdarkstyle.load_stylesheet())

    # Create main widget and horizontal layout
    main_widget = QWidget()
    layout = QHBoxLayout(main_widget)

    # -------------------- Left panel: Controls --------------------
    # Create magicgui form for hologram computation
    gui = magicgui(
        compute_hologram,
        xoffA={"min": -300, "max": 300},
        yoffA={"min": -300, "max": 300},
        xoffB={"min": -300, "max": 300},
        yoffB={"min": -300, "max": 300},
        modulation_type={
            "choices": [
                "Pure X", "Pure Y", "Circular Left", "Circular Right",
                "Azimuthal Polarization", "Radial Polarization",
                "Laguerre-Gaussian Lin", "Laguerre-Gaussian Left",
                "Laguerre-Gaussian Right", "Zeros Hologram", "Sinus Hologram"
            ]
        },
        slm_correction_type={
            "choices": [
                "No Correction", "Direct Correction", "Double-Pass Correction"
            ]
        },
        auto_call=False,
        call_button="Compute Hologram",
    )

    # Create custom widgets for monitor selection and SLM display
    monitor_label = QLabel("Monitor Number:")
    monitor_spin = QSpinBox()
    monitor_spin.setMinimum(0)
    monitor_spin.setValue(2)

    display_checkbox = QCheckBox("Display Hologram on SLM")
    display_checkbox.setChecked(False)

    # Organize left panel layout
    control_layout = QVBoxLayout()
    control_layout.addWidget(gui.native)
    control_layout.addWidget(monitor_label)
    control_layout.addWidget(monitor_spin)
    control_layout.addWidget(display_checkbox)
    layout.addLayout(control_layout)

    # -------------------- Right panel: Matplotlib canvas --------------------
    # Configure matplotlib style
    plt.rcParams.update({
        "figure.facecolor": "#19232d",
        "axes.facecolor": "#2b2b3f",
        "axes.edgecolor": "#4f5b66",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": "#4f5b66",
    })

    # Create figure and canvas
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)

    # Add toolbar and canvas in a vertical layout
    right_layout = QVBoxLayout()
    toolbar = NavigationToolbar(canvas, main_widget)
    right_layout.addWidget(toolbar)
    right_layout.addWidget(canvas)
    layout.addLayout(right_layout, stretch=1)

    # -------------------- SLM Handling --------------------
    slm = None  # Reference to the SLM object
    slm_created = False  # Flag to indicate first-time SLM creation

    # Callback function for updating image and SLM
    @gui.called.connect
    def update_image(result):
        nonlocal slm, slm_created
        if result is None:
            return

        hologram = result[0]

        # Update matplotlib canvas
        ax.clear()
        ax.imshow(hologram, cmap="gray", origin="upper", vmin=0, vmax=255)
        ax.axis("off")
        canvas.draw()

        # Handle first tick on display_checkbox: create SLM
        if display_checkbox.isChecked() and not slm_created:
            from qtpy.QtCore import QTimer

            def create_slm():
                nonlocal slm, slm_created
                monitor = monitor_spin.value() - 1  # Zero-indexed
                slm = slmpy.SLMdisplay(isImageLock=False, monitor=monitor)
                slm.updateArray(hologram.astype("uint8"))
                slm_created = True

                # Freeze checkbox and monitor selection to avoid wxPython threading issues
                display_checkbox.setEnabled(False)
                monitor_spin.setEnabled(False)

            # Ensure SLM creation runs on main Qt thread
            QTimer.singleShot(0, create_slm)

        # Update existing SLM if already created
        elif slm_created:
            slm.updateArray(hologram.astype("uint8"))

    # Compute hologram at startup
    gui()

    # -------------------- Finalize main window --------------------
    main_widget.setWindowTitle("Hologram GUI")
    main_widget.resize(1200, 800)
    main_widget.show()

    # Start Qt event loop
    app.exec()


if __name__ == "__main__":
    main()
