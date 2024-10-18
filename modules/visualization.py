from PIL import ImageDraw

def visualize_detections(image, detections, errors):
    """
    Visualizes detections and highlights errors on the image.

    Args:
        image (PIL.Image.Image): The original image.
        detections (List[dict]): List of detected signs with bounding boxes and codes.
        errors (List[dict]): List of rule violations.
    """
    draw = ImageDraw.Draw(image)
    
    # Draw detections
    for det in detections:
        bbox = det['bbox']
        draw.rectangle(bbox, outline='green', width=2)
        if 'code' in det and det['code']:
            draw.text((bbox[0], bbox[1] - 10), det['code'], fill='green')
    
    # Highlight errors
    for error in errors:
        sign = error['sign']
        rule = error['rule_violated']
        bbox = sign['bbox']
        draw.rectangle(bbox, outline='red', width=3)
        draw.text((bbox[0], bbox[1] - 20), rule, fill='red')