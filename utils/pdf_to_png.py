import fitz
import os

def convert_pdf_to_png(pdf_path, output_dir, zoom=2.0):
    """
    Converts each page of a PDF to a high-resolution PNG image.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the converted PNG images.
        zoom (float): Zoom factor for resolution scaling.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        output_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(output_path)
        print(f"Saved {output_path}")
    doc.close()

# Example usage
if __name__ == "__main__":
    pdf_path = 'data/raw/cad_pdfs/input.pdf'
    output_dir = 'data/processed/pngs/'
    convert_pdf_to_png(pdf_path, output_dir)