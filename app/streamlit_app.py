import streamlit as st
import os
from PIL import Image, ImageDraw
from config.logging_config import logger
from modules.object_detection import perform_detection
from modules.text_recognition import extract_text
from modules.text_association import associate_text_with_signs
from modules.knowledge_base import initialize_knowledge_base, validate_detections
from modules.visualization import visualize_detections

def main():
    st.title("CAD Draft Reviewing System")
    
    uploaded_file = st.file_uploader("Upload CAD PDF", type=["pdf"])
    if uploaded_file is not None:
        # Save uploaded PDF
        with open("data/raw/cad_pdfs/uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info("Uploaded PDF saved")
        
        # Convert PDF to PNG
        from utils.pdf_to_png import convert_pdf_to_png
        convert_pdf_to_png("data/raw/cad_pdfs/uploaded.pdf", "data/processed/pngs/")
        
        logger.info("PDF converted to PNG")
        
        # Tile Images
        from utils.image_tiling import tile_image
        png_files = os.listdir("data/processed/pngs/")
        for png in png_files:
            image_path = os.path.join("data/processed/pngs/", png)
            tiles_dir = os.path.join("data/processed/tiles/", os.path.splitext(png)[0])
            tile_image(image_path, tiles_dir)
        
        # Perform Object Detection on Tiles
        detections = []
        from modules.object_detection import detect_objects_in_tiles
        for png in png_files:
            tiles_dir = os.path.join("data/processed/tiles/", os.path.splitext(png)[0])
            tile_paths = [os.path.join(tiles_dir, tile) for tile in os.listdir(tiles_dir)]
            tile_detections = detect_objects_in_tiles(tile_paths)
            detections.extend(tile_detections)
        
        logger.info("Object detection completed")
        
        # Perform OCR on Tiles
        ocr_results = []
        from modules.text_recognition import extract_text
        for tile in detections:
            ocr = extract_text(tile['tile_path'])
            ocr_results.extend(ocr)
        
        logger.info("OCR completed")
        
        # Associate Text with Signs
        from modules.text_association import associate_text_with_signs
        detections = associate_text_with_signs(detections, ocr_results)
        
        logger.info("Text association completed")
        
        # Initialize Knowledge Base
        from modules.knowledge_base import initialize_knowledge_base
        initialize_knowledge_base("data/annotations/rules.xlsx", source_type='excel')
        
        logger.info("Knowledge base initialized")
        
        # Validate Detections
        from modules.knowledge_base import validate_detections
        errors = validate_detections(detections)
        
        logger.info("Detections validated")
        
        # Visualize Detections and Errors
        image = Image.open(detections[0]['tile_path']).convert("RGB")
        visualize_detections(image, detections, errors)
        
        st.image(image, caption='CAD Draft with Detections and Errors', use_column_width=True)
        
        logger.info("Detections visualized")
        
        # Feedback Mechanism
        if st.button("Submit Feedback"):
            st.success("Feedback submitted successfully!")
            # Implement feedback handling logic

if __name__ == "__main__":
    main()
