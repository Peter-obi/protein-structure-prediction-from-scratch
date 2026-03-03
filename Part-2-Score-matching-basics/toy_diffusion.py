import torch
import torch.distributions as D
"""
mixture of Gaussians - 3 modes
"""

means = torch.tensor([-3.0, 0.0, 3.0])
stdev = torch.tensor([0.4, 0.4, 0.4])
weights = torch.ones(3)/3.0

categorical_dist = D.Categorical(probs=weights) # select the mode
component_dist = D.Normal(means, stdev)
gmm = D.MixtureSameFamily(categorical_dist, component_dist)
samples = gmm.sample((1000,)) 

betas = torch.linspace(0.0001, 0.02, 200)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
sqrt_aphas_cumprod = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_bar)

t = torch.zeros([128])
embed_dim = 64

def get_timestep_embedding(t, embed_dim):
  half_dim = embed_dim//2
  frequencies = torch.exp(torch.arange(0, half_dim)* -torch.log(torch.tensor(10000.0))/(half_dim - 1))
  args = t.unsqueeze(-1) * frequencies.unsqueeze(0)
  embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
  return embedding

class DiffusionModel(torch.nn.Module):
  def __init__(self):
    super(DiffusionModel, self).__init__()
    self.time_proj = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, embed_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(embed_dim, embed_dim)
        )
    self.data_proj = torch.nn.Sequential(
        torch.nn.Linear(1, embed_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(embed_dim, embed_dim)
        )
    self.hidden1 = torch.nn.Linear(embed_dim, 128)
    self.activation = torch.nn.SiLU()
    self.hidden2 = torch.nn.Linear(128, 128)
    self.hidden3 = torch.nn.Linear(128, 128)
    self.output = torch.nn.Linear(128, 1)

  def forward(self, t, x):
    t = get_timestep_embedding(t, embed_dim)
    t = self.time_proj(t)
    x = self.data_proj(x)
    merge_proj = t + x
    merge_proj = self.hidden1(merge_proj)
    merge_proj = self.activation(merge_proj)
    merge_proj = self.hidden2(merge_proj)
    merge_proj = self.activation(merge_proj)
    merge_proj = self.hidden3(merge_proj)
    merge_proj = self.activation(merge_proj)
    out = self.output(merge_proj)
    return out
