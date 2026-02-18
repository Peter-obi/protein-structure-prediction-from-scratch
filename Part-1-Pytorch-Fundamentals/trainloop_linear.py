import torch

DIM_IN = 1000
BATCH_SIZE = 16
DIM_OUT = 1
class TinyLinear(torch.nn.Module):

  def __init__(self):
    super(TinyLinear, self).__init__()

    self.linear1 = torch.nn.Linear(DIM_IN, 50)
    self.activation = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(50, DIM_OUT)

  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x

model = TinyLinear()

x = torch.randn(BATCH_SIZE, DIM_IN, dtype=torch.float32)
y = torch.randn(BATCH_SIZE, DIM_OUT, dtype=torch.float32)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
running_loss = 0.0
for epoch in range(20):
  optimizer.zero_grad()
  output = model(x)
  loss = criterion(output, y)
  loss.backward()
  optimizer.step()
  running_loss += loss.item()
  print(
      f"Epoch:{epoch+1:03d} "
      f"Loss:{loss:.2f}")
