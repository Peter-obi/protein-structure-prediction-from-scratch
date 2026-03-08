"""
1D VP-DIFFUSION TOY MODEL (GAUSSIAN MIXTURE)
Learn the score function of a simple Gaussian Mixture Model (GMM) via denoising score matching.
Define a 3-mode Gaussian Mixture:
    Means   = [-3.0, 0.0, 3.0]
    Std     = 0.4 (shared)
    Weights = uniform
    
Sampling process:
    1. Sample mode index  k ~ Categorical(π)
    2. Sample coordinate  x0 ~ N(μ_k, σ²)

This defines pdata(x), the distribution we want the model to learn.
Then use a linear beta schedule over T = 200 steps:
    β_t ∈ [1e-4, 0.02]
    α_t = 1 - β_t
    ᾱ_t = ∏_{i=1}^t α_i

Closed-form noisy sampling:
    x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
    ε ~ N(0, I)

This allows O(1) sampling of any timestep without recursive simulation.

As t → T, x_t approaches N(0, I).

The network learns:

    ε_θ(x_t, t)

Architecture design:

• Sinusoidal timestep embedding (dim=64)
    Encodes scalar timestep into high-frequency + low-frequency components
    using sin/cos positional encoding.

• Dual-stream projection:
    - Data projector:  x_t (1D) → 64D
    - Time projector:  embedding → 64D

• Feature fusion:
    fused = data_features + time_features

• MLP backbone:
    64 → 128 → 128 → 128 → 1
    Activation: SiLU (smooth, score-friendly)

The output is a scalar noise prediction ε̂.

Denoising Score Matching (MSE on noise):

    min_θ  E_{t,x0,ε} [ || ε - ε_θ(x_t, t) ||² ]

Under the VP parameterization, this is equivalent to learning the
time-dependent score ∇_x log p_t(x).

TENSOR SHAPES to track

x_t            : [Batch, 1]
t              : [Batch]
time embedding : [Batch, 64]
hidden layers  : [Batch, 128]
output         : [Batch, 1]

Minimal, fully interpretable sandbox for understanding:

• Forward noising
• Closed-form diffusion sampling
• Sinusoidal time conditioning
• Noise prediction parameterization
• Score learning in 1D
"""
import torch
import torch.distributions as D

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
      
model = DiffusionModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
