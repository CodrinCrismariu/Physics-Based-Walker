import torch
import numpy as np
import math

size_x, size_y = 1.5, 1.0
res = 0.1
num_x = len(torch.arange(-size_x / 2, size_x / 2 + res * 0.5, res))
num_y = len(torch.arange(-size_y / 2, size_y / 2 + res * 0.5, res))
print(num_x, num_y, num_x * num_y)
