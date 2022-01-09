import torch

def init_gpu(use_gpu=True, gpu_id=0, verbose=False) -> torch.device:
    """Initialize device. ('cuda:0' or 'cpu')"""
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        if verbose:
            print(f"Using GPU id {gpu_id}.\n")
    else:
        device = torch.device("cpu")
        if not torch.cuda.is_available():
            print("GPU not detected. Defaulting to CPU.\n")
        elif not use_gpu:
            if verbose:
                print("Device: set to use CPU.\n")
    return device