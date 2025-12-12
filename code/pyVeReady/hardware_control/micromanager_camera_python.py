from pymmcore_plus import CMMCorePlus
import matplotlib.pyplot as plt


class PyMicroManagerPlusCamera:
    def __init__(self, camera_type):
        self.camera_type = camera_type
        self.core = CMMCorePlus.instance()
        self.config_file = self._get_config_file()

    def _get_config_file(self):
        config_files = {
            'TIS Camera': "Imaging_Source_Cam.cfg",
            'Other Camera': "Other_Cam.cfg"  # Add more mappings as needed
        }
        if self.camera_type not in config_files:
            raise ValueError("Unknown camera type")
        return config_files[self.camera_type]

    def connect(self):
        self.core.loadSystemConfiguration(self.config_file)
        print(f'{self.camera_type} connected successfully')

    def disconnect(self):
        self.core.unloadAllDevices()
        print(f'{self.camera_type} disconnected successfully')

    def get_exposure(self):
        return self.core.getExposure()

    def set_exposure(self, exposure_ms):
        self.core.setExposure(exposure_ms)
        self.core.setExposure(exposure_ms)

    def snap(self):
        self.core.snapImage()
        img = self.core.getImage()
        return img

    def set_exposure_interactive(self):
        plt.ion()
        fig, ax = plt.subplots()
        while True:
            exposure = self.get_exposure()
            img = self.snap()

            ax.clear()
            ax.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
            ax.set_title(f'Exposure: {exposure} ms')
            plt.draw()
            plt.pause(0.1)

            new_exposure = input("Enter new exposure time (ms) or press Enter to keep current: ")
            if new_exposure.strip():
                try:
                    self.set_exposure(float(new_exposure))
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
            else:
                break
        plt.ioff()
        plt.close()

    def get_pixel_size_um(self):
        pixel_sizes_um = {
            'TIS Camera': 6,
            'Other Camera': 6  # Add more mappings as needed
        }
        return pixel_sizes_um[self.camera_type]

if __name__ == '__main__':
    camera = PyMicroManagerPlusCamera('TIS Camera')
    camera.connect()
