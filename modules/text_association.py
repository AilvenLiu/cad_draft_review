import math

def associate_text_with_signs(detections: list, ocr_results: list, max_distance=50) -> list:
    """
    Associates OCR-extracted texts with detected signs based on proximity.

    Args:
        detections (List[dict]): List of detected signs with bounding boxes.
        ocr_results (List[dict]): List of extracted texts with bounding boxes.
        max_distance (int): Maximum distance to consider for association.

    Returns:
        List[dict]: Detections augmented with associated text.
    """
    for sign in detections:
        sign_center = get_center(sign['bbox'])
        nearest_text = None
        min_distance = float('inf')
        
        for text in ocr_results:
            text_center = get_center(text['bbox'])
            distance = compute_distance(sign_center, text_center)
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                nearest_text = text['text']
        
        sign['code'] = nearest_text
    return detections

def get_center(bbox: list) -> tuple:
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

def compute_distance(center1: tuple, center2: tuple) -> float:
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)