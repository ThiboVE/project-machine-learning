import torch

device = "cuda"

x = torch.rand(1000, 1000).to(device)
y = torch.mm(x, x)
print("GPU test OK, result:", y[0,0].item())

