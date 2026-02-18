import torch

x = torch.rand(4, 4, requires_grad=True)
loss = torch.sum(x)
y = loss.backward()
x.grad
