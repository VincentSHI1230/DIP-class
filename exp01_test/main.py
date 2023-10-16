import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
from pathlib import Path

print(torch.__version__)
print(Path.cwd())
