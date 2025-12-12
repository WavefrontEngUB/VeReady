from tqdm import tqdm
from pyVeReady.utils.image_utils import *


if __name__ == "__main__":
    all_data = load_hdf5('Select File')['HamaCAM']
    print('File Loaded Successfully')
    z_slices = 60

    N = all_data.shape[0]
    chunk_size = N // z_slices

    for i in tqdm(range(z_slices), desc="Saving slices"):
        start = i * chunk_size
        end = start + chunk_size
        tifffile.imwrite(f"stack_slice_{i:03d}.tif", all_data[start:end].astype(all_data.dtype))
