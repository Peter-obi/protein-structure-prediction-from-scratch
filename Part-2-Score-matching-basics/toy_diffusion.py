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
