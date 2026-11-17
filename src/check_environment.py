
import torch
import cv2
import numpy as np
import sklearn

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
