import os
from utils.pdf_to_png import convert_pdf_to_png

def main():
    pdf_dir = 'data/raw/cad_pdfs/'
    png_dir = 'data/processed/pngs/'
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            output_subdir = os.path.join(png_dir, os.path.splitext(pdf_file)[0])
            convert_pdf_to_png(pdf_path, output_subdir)

if __name__ == "__main__":
    main()