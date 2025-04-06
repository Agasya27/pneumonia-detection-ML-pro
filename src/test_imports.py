import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("OpenCV version:", cv2.__version__)
print("PIL version:", Image.__version__)

# Test PyTorch
device = torch.device("cpu")
print("\nUsing device:", device)

# Test CUDA availability
print("CUDA available:", torch.cuda.is_available())

print("\nAll imports successful!") 