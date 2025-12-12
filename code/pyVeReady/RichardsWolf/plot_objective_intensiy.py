from pyVeReady.utils.image_utils import *
import tifffile

# Folder with the .tif files
folder = r'C:\Users\laboratori\OneDrive - Universitat de Barcelona\PhD\VeReady\Measures\objective_pupil_measure'  # change this to your actual folder path

# Load all .tif images and stack them
image_stack = []
filenames = []

for filename in os.listdir(folder):
    if filename.endswith('.tif'):
        try:
            param = float(os.path.splitext(filename)[0])  # filename is like "100.tif"
            img = tifffile.imread(os.path.join(folder, filename))
            image_stack.append(img)
            filenames.append((param, filename))
        except:
            print(f"Skipping {filename} (invalid format or read error)")

# Convert to array
image_stack = np.array(image_stack)
sum_image = image_stack.sum(axis=0)

# Interactive cropping
plt.imshow(image_stack[0,:,:], cmap='gray')
plt.title("Click top-left and bottom-right corners to crop")
pts = plt.ginput(2)
plt.close()

# Get crop coordinates
(x1, y1), (x2, y2) = pts
x1, x2 = int(round(min(x1, x2))), int(round(max(x1, x2)))
y1, y2 = int(round(min(y1, y2))), int(round(max(y1, y2)))

# Crop and compute total intensity per image
data = []
for i, (param, filename) in enumerate(filenames):
    img = image_stack[i][y1:y2, x1:x2]
    total_intensity = img.sum()
    data.append((param, total_intensity))

# Sort and plot
data.sort()
params, intensities = zip(*data)

plt.figure(figsize=(6, 4))
plt.plot(params, intensities, marker='o')
plt.xlabel('Parameter')
plt.ylabel('Total Intensity (cropped)')
plt.title('Cropped Intensity vs. Parameter')
plt.grid(True)
plt.tight_layout()
plt.show()

viewer = Imshow3D(image_stack)