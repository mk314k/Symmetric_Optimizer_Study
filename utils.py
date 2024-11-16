import torch

def device():
    """
    Determines the best available device: CUDA, MPS, or CPU.

    Returns:
        torch.device: The most appropriate device for computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

