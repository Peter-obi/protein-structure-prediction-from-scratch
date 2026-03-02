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
samples.shape

timesteps = 200
betas = torch.linspace(0.0001, 0.02, 200)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
sqrt_aphas_cumprod = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_bar)
