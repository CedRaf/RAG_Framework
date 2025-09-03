import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# See how many GPUs PyTorch can see
print("GPU count:", torch.cuda.device_count())

# Name of the first GPU
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))