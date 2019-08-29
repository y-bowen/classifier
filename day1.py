from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
# Addition: providing an output tensor as argument
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)