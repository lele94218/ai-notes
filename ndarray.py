import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)
c = a + b


C = c.numpy()
print(c)
print(C)


e = torch.arange(10).reshape((5, 2))
print(e)
