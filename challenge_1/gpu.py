### This file is used to check if the GPU is available and to print its details. It can be run independently to verify the GPU setup before running the main training script.

import torch

print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("CUDA Version:", torch.version.cuda)