from pyVeReady.utils.image_utils import *
from pyVeReady.utils.paths_utils import *
from tqdm import tqdm
from image_processing_functions import *

directory = ask_directory_location('Select Directory')
files = sorted(f for f in os.listdir(directory) if f.endswith(".tif"))

moving_integration_roi = True

reconstructed_slices = []
for fname in tqdm(files, desc="Processing TIFF files"):
    path = os.path.join(directory, fname)
    data = tifffile.imread(path)

    data, _ = subtract_background(data)
    if moving_integration_roi:
        data, _ = extract_centered_rois(data, 9)

    widefield_image = compute_integrated_image(data)
    reconstructed_slices.append(widefield_image)

reconstructed_slices = np.array(reconstructed_slices, dtype=np.float64)
viewer = Imshow3D(reconstructed_slices)

tifffile.imwrite('Reconstructed_Volume.tif', reconstructed_slices.astype(np.float32))