import torch

# Scale
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x**y)


# Vector
x = torch.arange(3)
print(x)
print(x[2])

# Tensors
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(a + X, (a * X).shape)

# Sum
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)

print(A)
print(A.sum())
print(A.sum(axis=0))
print(A.sum(axis=1))

B = torch.arange(24).reshape(2,3,4) # shape=(2,3,4)
print(B)
print(B.sum(axis=0))   # (3,4) → 在第0维求和
print(B.sum(axis=1))   # (2,4) → 在第1维求和
print(B.sum(axis=2))   # (2,3) → 在第2维求和


A = torch.arange(2*3*4*5).reshape(2,3,4,5)  # shape: (N=2, C=3, H=4, W=5)

print("原始形状:", A)

print("axis=0 →", A.sum(axis=0))  # (3,4,5)
print("axis=1 →", A.sum(axis=1))  # (2,4,5)
print("axis=2 →", A.sum(axis=2))  # (2,3,5)
print("axis=3 →", A.sum(axis=3))  # (2,3,4)

# Multiplication

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = torch.ones(3, 4)

print("Multiplication: ", A, B, torch.mm(A, B), A@B)


# Norm
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
print(torch.abs(u))
print(torch.abs(u).sum())

u = torch.ones((4, 9))
print(u)
print(torch.norm(u))
