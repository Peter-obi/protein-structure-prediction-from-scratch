import torch
import math


BATCH_SIZE = 32
SEQ_LENGTH = 512
EMBED_DIM = 256
HEADS = 8

class MultiHeadAttention(torch.nn.Module):
  def __init__(self):
    super(MultiHeadAttention, self).__init__()
    self.head_dim = EMBED_DIM//8
    self.q_proj = torch.nn.Linear(EMBED_DIM, EMBED_DIM)
    self.k_proj = torch.nn.Linear(EMBED_DIM, EMBED_DIM)
    self.v_proj = torch.nn.Linear(EMBED_DIM, EMBED_DIM)
    self.dropout = torch.nn.Dropout()
    self.softmax = torch.nn.Softmax(dim=-1)
    self.out_proj = torch.nn.Linear(EMBED_DIM, EMBED_DIM)

  def forward(self, x):
    q = self.q_proj(x)
    q_split = q.view(BATCH_SIZE, SEQ_LENGTH, HEADS, self.head_dim)
    q = q_split.transpose(1,2)
    k = self.k_proj(x)
    k_split = k.view(BATCH_SIZE, SEQ_LENGTH, HEADS, self.head_dim)
    k = k_split.transpose(1,2)
    v = self.v_proj(x)
    v_split = v.view(BATCH_SIZE, SEQ_LENGTH, HEADS, self.head_dim)
    v = v_split.transpose(1, 2)
    attn_1 = self. dropout(self.softmax((torch.einsum('bhid,bhjd->bhij', q, k)) / math.sqrt(self.head_dim))) # divide to not make values big and softmax to convert scores to probabilities
    attn_2 = torch.einsum('bhij,bhjd->bhid', attn_1, v)
    attn_2_transposed = attn_2.transpose(1,2)
    attn_2 = attn_2_transposed.reshape(BATCH_SIZE, SEQ_LENGTH, EMBED_DIM)  #memory is no longer continguous so use reshape not view
    x = self.out_proj(attn_2)
    return x

class TransformerWrapped(torch.nn.Module):
  def __init__(self):
    super(TransformerWrapped, self).__init__()
    self.layernorm1 = torch.nn.LayerNorm(EMBED_DIM) #LayerNorm keeps track of two learnable parameters (γ and β) for every feature in your embedding to help the model scale the data after normalization.
    self.attn = MultiHeadAttention()
    self.dropout = torch.nn.Dropout()
    self.layernorm2 = torch.nn.LayerNorm(EMBED_DIM)
    self.mlp = torch.nn.Sequential(
        torch.nn.Linear(EMBED_DIM, EMBED_DIM * 4),
        torch.nn.Dropout(),
        torch.nn.ReLU(),
        torch.nn.Linear(EMBED_DIM * 4, EMBED_DIM),
        torch.nn.Dropout()
    )

  def forward(self, x):
    residual = x
    x = self.layernorm1(x)
    x = self.attn(x)
    x = self.dropout(x)
    x = residual + x
    residual = x
    x = self.layernorm2(x)
    x = self.mlp(x)
    x = residual + x
    return x
    
model = TransformerWrapped()
x = torch.randn(32, 512, 256)
y = torch.randn(32, 512, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
running_loss = 0.0

for epoch in range(20):
  optimizer.zero_grad()
  output = model(x)
  loss = criterion(output, y)
  loss.backward()
  optimizer.step()
  running_loss += loss.item()
  print(f"Epcoh: {epoch+1:03d} "
        f"Loss: {loss:.2f}")
