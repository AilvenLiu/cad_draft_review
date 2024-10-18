import unittest
from modules.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        pdf_dir = 'data/raw/cad_pdfs/test.pdf'
        png_dir = 'data/processed/pngs/test_pngs/'
        tiles_dir = 'data/processed/tiles/test_tiles/'
        transform = None  # or define a mock transform

        result = preprocess_data(pdf_dir, png_dir, tiles_dir, transform)
        self.assertIsNone(result)  # Assuming function returns None

if __name__ == '__main__':
    unittest.main()