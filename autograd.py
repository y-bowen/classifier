from __future__ import print_function
import torch

# x = torch.ones(2, 2, requires_grad=True)
# print(x)

# y = x + 2
# print(y)

# print(y.grad_fn)

# z = y * y * 3
# out = z.mean()

# print(z, out)

# out.backward()

# print(x.grad)


x = torch.randn(3, requires_grad=True)

y = x * 2

print(y.data.norm())
while y.data.norm() < 1000:
    y = y * 2
print(y.data.norm())
print(y)