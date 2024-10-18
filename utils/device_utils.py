import torch

def get_device():
    """
    Determine the appropriate device for computation.
    Returns:
        torch.device: The device to use for computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Usage example:
# device = get_device()
# print(f"Using device: {device}")
