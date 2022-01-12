import torch.nn as nn
import torch

x = torch.zeros((64, 40, 28, 28))

l = nn.ConvTranspose2d(40, 40, 2, 2)

print(l(x).shape)

a = [1, 2, 3, 4]
print(a[::-1])

