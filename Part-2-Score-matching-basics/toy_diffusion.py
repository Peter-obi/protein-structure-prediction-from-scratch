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
