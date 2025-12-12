from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import numpy as np
import svgwrite
import io
from tqdm import tqdm
from physical_constants import MM, PDF_SECTOR_LENGTH, PRINTED_HOLOGRAM_SIZE


def mm_to_points(length_mm):
    """Convert millimeters to points.

    1 point = 1/72 inch, 1 inch = 25.4 mm.

    Args:
        length_mm (float): Length in millimeters.

    Returns:
        float: Length in points.
    """
    return length_mm * 2.83465


def array_img_to_svg(image_array):
    """Convert a 2D grayscale numpy array into an SVG image.

    Args:
        image_array (np.ndarray): A 2D numpy array representing a grayscale image (0-255 values).

    Returns:
        str: SVG string representation of the image.
    """
    height, width = image_array.shape
    dwg = svgwrite.Drawing(size=(width, height))
    for y in range(height):
        for x in range(width):
            gray = int(image_array[y, x])
            color = svgwrite.utils.rgb(gray, gray, gray, mode='RGB')
            dwg.add(dwg.rect(insert=(x, y), size=(1, 1), fill=color))
    svg_string = io.StringIO()
    dwg.write(svg_string)
    return svg_string.getvalue()


def create_empty_pdf(output_pdf):
    """Create an empty A4-sized PDF and return the canvas object.

    Args:
        output_pdf (str): The file path where the PDF will be saved.

    Returns:
        reportlab.pdfgen.canvas.Canvas: The PDF canvas object.
    """
    pdf_canvas = canvas.Canvas(output_pdf, pagesize=A4)
    return pdf_canvas


def place_svg_in_pdf(pdf_canvas, image_array, sector_length_mm, image_size_mm, row, col, label = None, offset = 0):
    """Place an SVG image at the center of a specified sector in an A4 PDF, optionally adding a label above the image.

    Args:
        pdf_canvas (reportlab.pdfgen.canvas.Canvas): The PDF canvas object.
        image_array (np.ndarray): A 2D numpy array representing a grayscale image.
        sector_length_mm (float): Size of each square sector in millimeters.
        image_size_mm (float): Desired size of the image inside the sector in millimeters.
        row (int): The row index of the sector in the grid.
        col (int): The column index of the sector in the grid.
        label (str, optional): A string to be placed above the image. Defaults to None.
        offset (int, optional): The position offset (in points) of the sector. Defaults to 0.
    """
    width, height = A4  # A4 size in points
    sector_length = mm_to_points(sector_length_mm)
    image_size = mm_to_points(image_size_mm)

    svg_data = array_img_to_svg(image_array)
    svg_bytes = io.BytesIO(svg_data.encode('utf-8'))
    drawing = svg2rlg(svg_bytes)

    # Calculate sector position
    x_center = (col + 0.5) * sector_length + offset
    y_top = height - row * sector_length - offset  # Top of the sector

    # Scale SVG to fit specified image size
    scale = image_size / max(drawing.width, drawing.height)
    drawing.width *= scale
    drawing.height *= scale
    drawing.scale(scale, scale)

    # Determine positions
    text_padding = mm_to_points(2)  # Padding for text
    text_y_position = y_top - text_padding  # Place text near the top of the sector
    image_y_position = text_y_position - drawing.height - text_padding  # Position image below text

    if label:
        font_size = sector_length * 0.085  # 8.5% of sector size
        pdf_canvas.setFont("Helvetica", font_size)
        pdf_canvas.drawCentredString(x_center, text_y_position, label)

    # Draw SVG in the PDF
    renderPDF.draw(drawing, pdf_canvas, x_center - drawing.width / 2, image_y_position)


def save_pdf(pdf_canvas, output_pdf):
    """Save the PDF to the specified file path.

    Args:
        pdf_canvas (reportlab.pdfgen.canvas.Canvas): The PDF canvas object.
        output_pdf (str): The file path where the PDF will be saved.
    """
    pdf_canvas.save()
    print(f"PDF saved as {output_pdf}")

if __name__ == "__main__":
    # Example usage
    pdf = create_empty_pdf("output.pdf")
    rows, cols = int(A4[1] // mm_to_points(PDF_SECTOR_LENGTH / MM)), int(A4[0] // mm_to_points(PDF_SECTOR_LENGTH / MM))

    total_sectors = rows * cols
    with tqdm(total=total_sectors, desc="Processing Sectors") as pbar:
        for row in range(rows):
            for col in range(cols):
                random_image = np.random.randint(0, 255, (30, 30), dtype = np.uint8)
                label = f"({row}, {col})"  # Example label
                place_svg_in_pdf(pdf, random_image, PDF_SECTOR_LENGTH / MM, PRINTED_HOLOGRAM_SIZE / MM, row, col, label)
                pbar.update(1)

    save_pdf(pdf, "output.pdf")