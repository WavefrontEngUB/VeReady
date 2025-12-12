import h5py
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt
import numpy as np

class HDF5DropWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setAcceptDrops(True)
        layout = QVBoxLayout()
        self.label = QLabel("Drag and drop an HDF5 file here")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().endswith((".h5", ".hdf5")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.endswith((".h5", ".hdf5")):
                self.load_hdf5(file_path)

    def load_hdf5(self, file_path):
        with h5py.File(file_path, "r") as f:
            data_dict = self.recursively_load(f)

        # Add first dataset to Napari
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                self.viewer.add_image(value, name=key)
                break

    def recursively_load(self, h5obj):
        data = {}
        for key in h5obj:
            item = h5obj[key]
            if isinstance(item, h5py.Group):
                data[key] = self.recursively_load(item)
            elif isinstance(item, h5py.Dataset):
                data[key] = item[()]
        return data