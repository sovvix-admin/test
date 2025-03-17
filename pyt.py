import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# Check the number of GPUs available
print(torch.cuda.device_count())

# Check the name of the GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
