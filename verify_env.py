import torch
import torchvision
import timm
import albumentations as A
import cv2
import numpy as np
import torchmetrics
import lightning
import wandb
import hydra
import optuna

print("Environment setup verification:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"torchvision version: {torchvision.__version__}")
print(f"timm version: {timm.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"torchmetrics version: {torchmetrics.__version__}")
print(f"lightning version: {lightning.__version__}")
print(f"Optuna version: {optuna.__version__}")
print("\nAll required packages are installed successfully!") 