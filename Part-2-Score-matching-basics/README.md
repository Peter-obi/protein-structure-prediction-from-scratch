## Diffusion / Score Matching Basics

**Goal:** Understand forward noising, reverse denoising, and what the model is predicting (ε / x₀ / v).

### Do 
- Read: Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*  
  https://arxiv.org/abs/2011.13456
- Optional: inspect the reference PyTorch repo to ground training/sampling mechanics:  
  https://github.com/yang-song/score_sde_pytorch

### Task
Implement a **1D diffusion toy**:
- **Data:** mixture of Gaussians (e.g., 2–8 modes)
- **Model:** ε-parameterization denoiser `ε_θ(x_t, t)`
- **Training:** sample `t`, noise `x₀ → x_t`, minimize `||ε_θ - ε||²`
- **Sampling:** simple DDPM-like reverse loop from `x_T ~ N(0, I)` down to `x_0`
- **Pass criteria:** generated samples match the mixture’s modes (histogram/KS check)
