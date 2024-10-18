import os
from utils.pdf_to_png import convert_pdf_to_png
from utils.image_tiling import tile_image
from torchvision import transforms

def preprocess_data(pdf_dir, png_dir, tiles_dir, transform=None):
    """
    Executes the complete data preprocessing pipeline:
    1. Converts PDFs to PNGs.
    2. Tiles large PNG images.
    3. Applies transformations.

    Args:
        pdf_dir (str): Directory containing raw CAD PDF files.
        png_dir (str): Directory to save converted PNG images.
        tiles_dir (str): Directory to save tiled images.
        transform (callable, optional): Transformations to apply to images.
    """
    # Step 1: Convert PDFs to PNGs
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            output_subdir = os.path.join(png_dir, os.path.splitext(pdf_file)[0])
            convert_pdf_to_png(pdf_path, output_subdir)
    
    # Step 2: Tile Images
    for png_subdir in os.listdir(png_dir):
        image_path = os.path.join(png_dir, png_subdir, f'page_1.png')  # Adjust as needed
        output_tiles_dir = os.path.join(tiles_dir, png_subdir)
        tile_image(image_path, output_tiles_dir)
    
    # Step 3: Apply Transformations (if any)
    if transform:
        # Implement transformation application logic
        pass

if __name__ == "__main__":
    pdf_dir = 'data/raw/cad_pdfs/'
    png_dir = 'data/processed/pngs/'
    tiles_dir = 'data/processed/tiles/'
    preprocess_data(pdf_dir, png_dir, tiles_dir)