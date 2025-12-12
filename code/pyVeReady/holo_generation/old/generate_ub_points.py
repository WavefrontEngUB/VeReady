import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('TkAgg')

def create_ub_image(height=100, width=100, font_size=50, row_spacing=10, col_spacing=10):
    # Create a blank image with black background
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    # Define font (default PIL font is used as a fallback)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Define position for the text
    text = "UB"
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    position = ((width - text_width) // 2, (height - text_height) // 2)

    # Draw the text in white (255)
    draw.text(position, text, fill=255, font=font)

    # Convert image to numpy array
    img_array = np.array(img)

    # Create a sparse dot representation (1 for scattered points in text, 0 for background)
    sparse_img = np.zeros_like(img_array)

    # Create a grid of points with defined separation in both rows and columns
    for y in range(0, height, row_spacing):
        for x in range(0, width, col_spacing):
            # Only place a point where the image has a white pixel (i.e., part of the text)
            if img_array[y, x] > 128:  # Check if the pixel is part of the text (white)
                sparse_img[y, x] = 1

    return sparse_img


if __name__ == "__main__":
    # Example usage
    sparse_image = create_ub_image(100, 100, 50, row_spacing=5, col_spacing=5)  # Adjust row and column spacing here
    plt.imshow(sparse_image, cmap="gray")
    plt.show()
