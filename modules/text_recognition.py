import easyocr

def extract_text(image_path: str) -> list:
    """
    Extracts text from an image using EasyOCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        List[dict]: List of extracted texts with their bounding boxes and confidence scores.
    """
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    ocr_results = []
    for bbox, text, confidence in results:
        x_min = min([point[0] for point in bbox])
        y_min = min([point[1] for point in bbox])
        x_max = max([point[0] for point in bbox])
        y_max = max([point[1] for point in bbox])
        ocr_results.append({
            'bbox': [x_min, y_min, x_max, y_max],
            'text': text,
            'confidence': confidence
        })
    return ocr_results