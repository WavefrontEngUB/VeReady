# from numpy.fft import fft2, ifft2, fftshift, ifftshift
# import os
# import cv2
# import tifffile
# from skimage.filters import threshold_otsu
# from skimage.measure import regionprops, label
#
# def load_tiff_images_to_list(filepaths, slicing_index):
#     """
#     Load images from a file or a list of files to a list. It only supports Tiff files.
#
#     Args:
#         filepaths (Union[str, List[str]]): A single file path, a list containing a single file path, or a list of multiple file paths.
#         slicing_index (int): If a single file path representing a 3D array is provided, this index determines
#                             the axis along which the array is sliced to obtain a list of 2D images.
#
#     Returns:
#         List[np.ndarray]: A list of 2D numpy arrays representing the loaded images.
#     """
#     loaded_images = []
#
#     if isinstance(filepaths, str) or (isinstance(filepaths, list) and len(filepaths) == 1):
#         # Handle single file case, either as a string or a list containing one path
#         if isinstance(filepaths, list):
#             filepaths = filepaths[0]
#
#         if not os.path.isfile(filepaths):
#             raise FileNotFoundError(f"The file {filepaths} does not exist.")
#
#         # Load the file and assume it is a 3D array
#         data = tifffile.imread(filepaths).astype(np.float64)
#
#         if len(data.shape) != 3:
#             raise ValueError("The file does not contain a 3D array.")
#
#         if slicing_index < 0 or slicing_index > 2:
#             raise ValueError(f"slicing_index {slicing_index} is invalid. It must be 0, 1, or 2.")
#
#         # Slice along the specified axis
#         loaded_images = [np.take(data, i, axis=slicing_index) for i in range(data.shape[slicing_index])]
#
#     elif isinstance(filepaths, list):
#         # Multiple files case
#         for filepath in filepaths:
#             if not os.path.isfile(filepath):
#                 raise FileNotFoundError(f"The file {filepath} does not exist.")
#
#             # Load the file and assume it is a 2D array
#             data = tifffile.imread(filepath).astype(np.float64)
#
#             if len(data.shape) != 2:
#                 raise ValueError(f"The file {filepath} does not contain a 2D array.")
#
#             loaded_images.append(data)
#
#     else:
#         raise TypeError("filepaths must be a string or a list of strings.")
#
#     return loaded_images
#
# def cross_correlation2D(array, reference):
#     ft_array = fftshift(fft2(ifftshift(array)))
#     ft_reference = fftshift(fft2(ifftshift(reference)))
#     ft_convolution = ft_array * np.conjugate(ft_reference)
#
#     # Inverse transform and find its maximum value
#     # Inverse FFT transform lacks a shift to obtain directly
#     # the indices of the maximum with respect to the center.
#     convolution = (ifft2(ifftshift(ft_convolution)))
#     ampl_convolution = np.real(np.conj(convolution) * convolution)
#
#     yloc_max, xloc_max = np.where(ampl_convolution == ampl_convolution.max())
#     loc_max = (-yloc_max[0], -xloc_max[0])
#
#     return loc_max
#
# def center_image_to_reference(image, reference=None, axis_index=None):
#     """
#     Aligns a single image, a list of images, or slices of a 3D array to a reference image
#     using cross-correlation. It does not support multi-channel images.
#
#     Parameters:
#         image (np.ndarray or list):
#             A single 2D image, a list of 2D images, or a 3D array of images.
#         reference (np.ndarray, optional):
#             The reference image for alignment. If None:
#             - For a single image, it is used as its own reference.
#             - For a list, the first image in the list is used as the reference.
#             - For a 3D array, the slice along `axis_index` at index 0 is used.
#         axis_index (int, optional):
#             If `image` is a 3D array, this specifies the slicing axis. Default is None.
#             Ignored for single images or lists.
#
#     Returns:
#         np.ndarray or list:
#             The aligned image(s), matching the structure of the input.
#     """
#
#     def align_single_image(img, ref):
#         location = cross_correlation2D(img, ref)
#         return np.roll(img, location, axis=(0, 1))
#
#     # Handle single 2D image
#     if isinstance(image, np.ndarray) and image.ndim == 2:
#         if reference is None:
#             reference = image  # Use the image itself as reference
#         return align_single_image(image, reference)
#
#     # Handle list of 2D images
#     if isinstance(image, list):
#         if reference is None:
#             reference = image[0]  # Use the first image in the list as reference
#         return [align_single_image(img, reference) for img in image]
#
#     # Handle 3D array
#     if isinstance(image, np.ndarray) and image.ndim == 3:
#         if axis_index is None:
#             raise ValueError("For 3D arrays, 'axis_index' must be specified.")
#         if reference is None:
#             reference = np.take(image, indices=0, axis=axis_index)  # First slice along the axis
#
#         # Align each slice along the specified axis
#         aligned_images = np.empty_like(image)
#         for idx in range(image.shape[axis_index]):
#             current_slice = np.take(image, indices=idx, axis=axis_index)
#             aligned_slice = align_single_image(current_slice, reference)
#             np.put_along_axis(aligned_images, np.expand_dims(idx, axis=0), aligned_slice, axis=axis_index)
#         return aligned_images
#
#     raise ValueError("Input must be a 2D image, a list of 2D images, or a 3D array.")
#
# def crop_array_centered(arr, row, col, size):
#     """
#     Crop a sub-array from a 2D array with the specified (row, col) as the center.
#
#     Parameters:
#         arr (ndarray): Input 2D array (image).
#         row (int): Center row of the crop.
#         col (int): Center column of the crop.
#         size (tuple): Tuple (height, width) of the crop size.
#
#     Returns:
#         ndarray: Cropped sub-array.
#     """
#     # Calculate the starting and ending indices for the crop
#     crop_height, crop_width = size
#     row_start = max(row - crop_height // 2, 0)
#     col_start = max(col - crop_width // 2, 0)
#     row_end = min(row_start + crop_height, arr.shape[0])
#     col_end = min(col_start + crop_width, arr.shape[1])
#
#     # Adjust start points if crop exceeds boundaries
#     row_start = min(row_start, row_end - crop_height)
#     col_start = min(col_start, col_end - crop_width)
#
#     return arr[row_start:row_end, col_start:col_end]
#
# def find_beam_hole_center(image_u8, roi_size = 50, show_estimation = False):
#     """
#     Finds the center of the hole in the beam from an image. If image is not given as U8
#     it performs the conversion to U8.
#
#     Parameters:
#         image_u8: 2D array U8 image.
#         roi_size: int, size of the Region of Interest (ROI) around the estimated center for refinement.
#
#     Returns:
#         (row, col): tuple of integers, the row and column indexes of the detected center.
#     """
#     # Convert to U8 image
#     image_u8 = np.uint8(image_u8)
#
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(image_u8, (5, 5), 0)
#
#     # Apply thresholding to isolate the beam
#     _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
#
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Find the largest contour (assumed to be the beam)
#     largest_contour = max(contours, key=cv2.contourArea)
#
#     # Compute the initial centroid of the beam
#     M = cv2.moments(largest_contour)
#     if M["m00"] != 0:
#         cX_initial = int(M["m10"] / M["m00"])
#         cY_initial = int(M["m01"] / M["m00"])
#     else:
#         cX_initial, cY_initial = 0, 0
#
#     # Define ROI around the initial center
#     x_start = max(cX_initial - roi_size, 0)
#     y_start = max(cY_initial - roi_size, 0)
#     x_end = min(cX_initial + roi_size, image_u8.shape[1])
#     y_end = min(cY_initial + roi_size, image_u8.shape[0])
#     roi = image_u8[y_start:y_end, x_start:x_end]
#
#     # Process the ROI to find the hole
#     roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
#     _, roi_thresh = cv2.threshold(roi_blurred, 50, 255, cv2.THRESH_BINARY_INV)
#
#     # Find contours in the ROI
#     roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Find the smallest contour (assumed to be the hole)
#     hole_contour = min(roi_contours, key=cv2.contourArea)
#
#     # Compute the centroid of the hole in the ROI
#     M_hole = cv2.moments(hole_contour)
#     if M_hole["m00"] != 0:
#         cX_hole = int(M_hole["m10"] / M_hole["m00"]) + x_start
#         cY_hole = int(M_hole["m01"] / M_hole["m00"]) + y_start
#     else:
#         cX_hole, cY_hole = cX_initial, cY_initial
#
#     if show_estimation:
#         plt.figure()
#         plt.imshow(image_u8, cmap ="gray")
#         plt.scatter(cX_hole, cY_hole, color = 'r')
#
#     return cY_hole, cX_hole
#
# def pixel_coordinates(array, decenter_x = 0, decenter_y = 0):
#     """
#     Computes the pixel coordinate grid for a given 2D array, optionally decentering the coordinates.
#
#     Parameters:
#         array (numpy.ndarray): Input 2D array (e.g., an image).
#         decenter_x (float): Horizontal offset for the center (default: 0).
#         decenter_y (float): Vertical offset for the center (default: 0).
#
#     Returns:
#         x_meshgrid (numpy.ndarray): 2D array representing the X Cartesian pixel coordinates of the meshgrid
#         y_meshgrid (numpy.ndarray): 2D arrays representing the Cartesian pixel coordinates of the meshgrid.
#         R, phi_meshgrid (numpy.ndarray): 2D arrays representing the polar pixel coordinates (radius and angle, respectively) of the meshgrid.
#
#     """
#     rows, cols = array.shape
#     rows_pixels = np.arange(rows, dtype = np.float64) - rows // 2 - decenter_y
#     cols_pixels = np.arange(cols, dtype = np.float64) - cols // 2 - decenter_x
#     x_meshgrid, y_meshgrid = np.meshgrid(cols_pixels, rows_pixels)
#     r_meshgrid, phi_meshgrid = np.sqrt(x_meshgrid ** 2 + y_meshgrid ** 2), np.arctan2(y_meshgrid, x_meshgrid)
#     return x_meshgrid, y_meshgrid, r_meshgrid, phi_meshgrid
#
#
# def split_array(array, split_fraction = 2):
#     """
#     Splits a 2D array into two parts along its larger dimension.
#     If rows and columns are equal, the array is split along columns.
#
#     Parameters:
#         array (numpy.ndarray): Input 2D array to split.
#         split_fraction (int): Division factor determining the split position (default: 2).
#
#     Returns:
#     - A, B (numpy.ndarray): The two split sections of the array.
#     - concat_ax (int): Axis along which the array was split (0 for rows, 1 for columns).
#     """
#     rows, cols = array.shape
#
#     if cols > rows:
#         A = array[:, :int(cols / split_fraction)]
#         B = array[:, int(cols / split_fraction):]
#         concat_ax = 1
#     elif rows > cols:
#         A = array[:int(rows / split_fraction), :]
#         B = array[int(rows / split_fraction):, :]
#         concat_ax = 0
#     else:
#         print("Rows and columns are equal; splitting along columns.")
#         A = array[:, :int(cols / split_fraction)]
#         B = array[:, int(cols / split_fraction):]
#         concat_ax = 1
#
#     return A, B, concat_ax
#
#
# def slice_offset_part(array, N, M, Xoff, Yoff):
#     # Calculate the starting indices with offsets
#     start_row = (array.shape[0] - N) // 2 + Yoff
#     start_col = (array.shape[1] - M) // 2 + Xoff
#
#     # Ensure the starting indices are within bounds
#     start_row = max(0, min(start_row, array.shape[0] - N))
#     start_col = max(0, min(start_col, array.shape[1] - M))
#
#     # Slice the desired part
#     offset_part = array[start_row:start_row + N, start_col:start_col + M]
#
#     return offset_part
#
# def pad2size(array, target_rows, target_cols):
#     """
#     Pads a 2D array to match the specified target size with constant values (zeros).
#     If the target dimensions are smaller than the input dimensions, no padding is added.
#
#     Parameters:
#         array (numpy.ndarray): Input 2D array to be padded.
#         target_rows (int): Desired number of rows in the output array.
#         target_cols (int): Desired number of columns in the output array.
#
#     Returns:
#         padded_array (numpy.ndarray): The padded array of shape (target_rows, target_cols).
#     """
#     input_rows, input_cols = array.shape
#
#     if target_rows < input_rows or target_cols < input_cols:
#         print("Target dimensions are smaller than the array size. No padding applied.")
#         return array
#
#     pad_top = (target_rows - input_rows) // 2
#     pad_bottom = target_rows - pad_top - input_rows
#
#     pad_left = (target_cols - input_cols) // 2
#     pad_right = target_cols - pad_left - input_cols
#
#     padded_array = np.pad(
#         array,
#         pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
#         mode='constant'
#     )
#
#     return padded_array
#
# def save_array_as_tiff(filename_out, saving_directory, array):
#     filename_out = filename_out.replace(' ', '_')
#     name, ext = os.path.splitext(filename_out)
#     if ext.lower() != '.tiff':
#         filename_out = f'{name}.tiff'
#
#     counter = 1
#     while os.path.exists(os.path.join(saving_directory, filename_out)):
#         filename_out = f'{name}_{counter}{ext}'
#         counter += 1
#
#     absolute_path = saving_directory + r'/' + filename_out
#     tifffile.imwrite(absolute_path, array)
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, RadioButtons
#
# class Imshow3D:
#     def __init__(self, volume):
#         """ MATLAB-like imshow3D function for visualizing 3D images. """
#         self.img = np.asarray(volume)
#         if self.img.ndim != 3:
#             raise ValueError("Input must be a 3D NumPy array.")
#
#         self.slice_axis = 2  # Default slicing axis: Z
#         self.update_slices()
#
#         # Initialize figure
#         self.fig, self.ax = plt.subplots()
#         plt.subplots_adjust(bottom=0.25, right=0.85)
#
#         self.im = self.ax.imshow(self.get_slice(), cmap='gray')
#         self.ax.set_title(self.get_title())
#
#         # Connect scroll event
#         self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
#
#         # Slider (Fully Purple)
#         ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lavender')  # Light purple background
#         self.slider = Slider(ax_slider, 'Slice', 0, self.slices - 1, valinit=self.current_slice, valfmt='%d',
#                              valstep=1)
#         self.slider.poly.set_color('purple')  # Make the slider bar purple
#         self.slider.valtext.set_color('purple')  # Make slider value text purple
#         self.slider.on_changed(self.on_slider_change)
#
#         # Radio buttons for slicing axis
#         ax_radio = plt.axes([0.88, 0.4, 0.1, 0.15], facecolor='lightgray')
#         self.radio = RadioButtons(ax_radio, ('X', 'Y', 'Z'))
#
#         # Set correct initial selection
#         self.radio.set_active(self.slice_axis)
#         self.radio.on_clicked(self.on_radio_change)
#
#         plt.show()
#
#     def update_slices(self):
#         """Update slice count and set current slice to middle."""
#         self.slices = self.img.shape[self.slice_axis]
#         self.current_slice = self.slices // 2
#
#     def get_slice(self):
#         """Extract the current slice based on the selected axis."""
#         if self.slice_axis == 0:
#             return self.img[self.current_slice, :, :]
#         elif self.slice_axis == 1:
#             return self.img[:, self.current_slice, :]
#         elif self.slice_axis == 2:
#             return self.img[:, :, self.current_slice]
#
#     def get_title(self):
#         """Generate title text with current slicing info."""
#         axis_labels = ['X', 'Y', 'Z']
#         return f"Axis: {axis_labels[self.slice_axis]} | Slice {self.current_slice + 1}/{self.slices}"
#
#     def update_slice(self):
#         """Update the displayed slice."""
#         self.ax.set_title(self.get_title())
#         self.im.set_array(self.get_slice())
#         self.fig.canvas.draw_idle()
#
#     def on_scroll(self, event):
#         """Scroll through slices using the mouse wheel."""
#         if event.step > 0:
#             self.current_slice = min(self.slices - 1, self.current_slice + 1)
#         else:
#             self.current_slice = max(0, self.current_slice - 1)
#         self.slider.set_val(self.current_slice)  # Sync slider
#         self.update_slice()
#
#     def on_slider_change(self, val):
#         """Update slice when slider is moved."""
#         self.current_slice = int(round(val))
#         self.update_slice()
#
#     def on_radio_change(self, label):
#         """Change slicing axis when radio button is clicked."""
#         axis_map = {'X': 0, 'Y': 1, 'Z': 2}
#         self.slice_axis = axis_map[label]
#         self.update_slices()
#
#         # Remove the old image and create a new one
#         self.im.remove()
#         self.im = self.ax.imshow(self.get_slice(), cmap='gray')
#
#         # Update slider limits dynamically
#         self.slider.valmin = 0
#         self.slider.valmax = self.slices - 1
#         self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
#         self.slider.set_val(self.current_slice)  # Reset slider position
#
#         self.update_slice()
#
# def interactive_cropper(array, axis = 0):
#     if not isinstance(array, np.ndarray) or array.ndim != 3:
#         raise ValueError(f"Input must be a 3D NumPy array, but got shape {array.shape}")
#
#     projection = np.max(array, axis=axis).astype(np.uint8)  # Max projection for visualization
#     roi = cv2.selectROI("Select ROI", projection, showCrosshair=True)
#     cv2.destroyAllWindows()
#
#     x, y, w, h = roi
#     if w == 0 or h == 0:
#         return None  # No valid selection
#
#     if axis == 0:
#         cropped_array = array[:, y:y + h, x:x + w]
#     elif axis == 1:
#         cropped_array = array[y:y + h, :, x:x + w]
#     elif axis == 2:
#         cropped_array = array[y:y + h, x:x + w, :]
#
#     return cropped_array
#
#
# def find_beam_center(image, thresholding='Otsu'):
#     """
#     Finds the center of the beam in an image by detecting the largest connected region.
#
#     Parameters:
#         image (np.ndarray): The input grayscale image containing the beam.
#         thresholding(str, optional): The method to obtain the threshold for labeling the image.
#     Returns:
#         tuple: (row, col) coordinates of the detected beam center.
#     """
#     if thresholding=='Otsu':
#         thresh = threshold_otsu(image)
#     else:
#         thresh = image.min() + 0.5 * (image.max() - image.min())
#
#     binary = image > thresh
#     labeled = label(binary)
#     regions = regionprops(labeled, image)
#     largest_region = max(regions, key = lambda r: r.area)
#     return largest_region.centroid
#
# def crop_image(image, row_center, col_center, roi_size):
#     row_center, col_center = int(row_center), int(col_center)
#     half_roi_size = roi_size // 2
#     cropped_im = image[row_center - half_roi_size : row_center + half_roi_size,
#                        col_center - half_roi_size : col_center + half_roi_size]
#     return cropped_im
#
#
# def make_collage(image_list, N):
#     """Creates an NxN collage from a list of images.
#
#     Args:
#         image_list (list of np.array): List of images as NumPy arrays.
#         N (int): Grid size (NxN).
#
#     Returns:
#         np.array: The final collage as an image array.
#     """
#     if len(image_list) != N ** 2:
#         raise ValueError("The number of images must be exactly N^2.")
#
#     # Get the size of the first image (assuming all are the same size)
#     h, w = image_list[0].shape
#
#     # Create an empty canvas for the collage
#     collage = np.zeros((N * h, N * w), dtype = np.float64)
#
#     # Fill the collage with images
#     for idx, img in enumerate(image_list):
#         row, col = divmod(idx, N)
#         collage[row * h:(row + 1) * h, col * w:(col + 1) * w] = img
#
#     return collage