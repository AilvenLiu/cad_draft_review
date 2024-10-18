import torch

def select_uncertain_samples(model, dataloader, device, threshold=0.5):
    """
    Selects samples with detection scores below the threshold for manual labeling.

    Args:
        model: Trained object detection model.
        dataloader: DataLoader for unlabeled data.
        device: Computation device.
        threshold (float): Confidence score threshold.

    Returns:
        List[dict]: List of uncertain detections.
    """
    model.eval()
    uncertain_samples = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Assume outputs contain 'scores'
            scores = outputs['scores']
            low_confidence = scores < threshold
            
            for img, lc in zip(images, low_confidence):
                if lc.any():
                    uncertain_samples.append(img.cpu())

    model.train()
    return uncertain_samples

def active_learning_step(model, dataloader, device, threshold=0.5):
    """
    Performs an active learning step by selecting uncertain samples.

    Args:
        model: Trained object detection model.
        dataloader: DataLoader for unlabeled data.
        device: Computation device.
        threshold (float): Confidence score threshold.

    Returns:
        List[torch.Tensor]: List of uncertain images.
    """
    return select_uncertain_samples(model, dataloader, device, threshold)
