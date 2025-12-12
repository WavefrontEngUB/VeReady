import matplotlib
from PyFilmHolo.pdf_functions import *
from PyFilmHolo.physical_constants import MM, UM, PDF_SECTOR_LENGTH, PRINTED_HOLOGRAM_SIZE
matplotlib.use('Tkagg')

def vertical_stripes(n_pixels):
    n_pixels = int(n_pixels)  # Ensure n_pixels is an integer
    row = np.array([0, 255] * ((n_pixels + 1) // 2))[:n_pixels]  # Create alternating pattern
    image = np.tile(row, (n_pixels, 1))  # Repeat the row to create a 2D array
    return image


min_pixel_size = 20 * UM
max_pixel_size = 500 * UM
n_gratings = 16
pixel_sizes = np.linspace(min_pixel_size, max_pixel_size, n_gratings)

# Create an empty PDF
pdf = create_empty_pdf("gratings_output.pdf")

# Define the grid for placement
rows = int(A4[1] // mm_to_points(PDF_SECTOR_LENGTH / MM))
cols = int(A4[0] // mm_to_points(PDF_SECTOR_LENGTH / MM))

# Generate and place gratings in the PDF
total_sectors = rows * cols
with tqdm(total = min(len(pixel_sizes), total_sectors), desc = "Placing Gratings") as pbar:
    count = 0  # Track how many gratings we have placed
    for row in range(rows):
        for col in range(cols):
            if count >= len(pixel_sizes):
                break  # Stop if we run out of gratings

            pixel_size = pixel_sizes[count]
            n_pixels = int(PRINTED_HOLOGRAM_SIZE // pixel_size)
            binary_grating = vertical_stripes(n_pixels)

            label = f"Grating Pixel: {pixel_size / UM:.1f} µm"  # Label in micrometers
            place_svg_in_pdf(pdf, binary_grating, PDF_SECTOR_LENGTH / MM, PRINTED_HOLOGRAM_SIZE / MM, row, col, label, offset = 15)

            count += 1
            pbar.update(1)

# Save the PDF
save_pdf(pdf, "gratings_output.pdf")