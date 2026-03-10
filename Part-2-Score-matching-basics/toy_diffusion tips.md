# 1D VP-Diffusion Toy Model (Gaussian Mixture)

Learn the score function of a simple Gaussian Mixture Model (GMM) via **denoising score matching**.

---

## Data Distribution

Define a 3-mode Gaussian Mixture:

| Parameter | Value |
|-----------|-------|
| Means     | `[-3.0, 0.0, 3.0]` |
| Std       | `0.4` (shared) |
| Weights   | uniform |

**Sampling process:**
1. Sample mode index $k \sim \text{Categorical}(\pi)$
2. Sample coordinate $x_0 \sim \mathcal{N}(\mu_k, \sigma^2)$

This defines $p_{\text{data}}(x)$, the distribution we want the model to learn.

---

## Noise Schedule (VP)

Linear beta schedule over $T = 200$ steps:

$$\beta_t \in [10^{-4},\ 0.02]$$

$$\alpha_t = 1 - \beta_t \qquad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$$

**Closed-form noisy sampling** (O(1), no recursive simulation):

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

> As $t \to T$, $x_t \to \mathcal{N}(0, I)$

---

## Network Architecture

The network learns: $\varepsilon_\theta(x_t, t)$
```
x_t (scalar) ──► Data Projector  (1 → 64) ──────┐
                                                  ├──► fused ──► MLP Backbone ──► ε̂ (scalar)
t (scalar)   ──► Sin/Cos Embed (→ 64) ──► Time Proj (64 → 64) ──┘
```

### Sinusoidal Timestep Embedding (dim = 64)
Encodes scalar timestep into high- and low-frequency components via sin/cos positional encoding.

### Dual-Stream Projection
- **Data projector:** $x_t\ (1\text{D}) \to 64\text{D}$
- **Time projector:** embedding $\to 64\text{D}$

### Feature Fusion
$$f_{\text{fused}} = f_{\text{data}} + f_{\text{time}}$$

### MLP Backbone
$$64 \to 128 \to 128 \to 128 \to 1$$

- **Activation:** SiLU (smooth, score-friendly)
- **Output:** scalar noise prediction $\hat{\varepsilon}$

---

## Training Objective — Denoising Score Matching

$$\min_\theta\ \mathbb{E}_{t,\, x_0,\, \varepsilon}\left[\, \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2 \,\right]$$

Under the VP parameterization, this is equivalent to learning the time-dependent score:

$$\nabla_x \log p_t(x)$$

---

## Tensor Shape Reference

| Tensor | Shape |
|--------|-------|
| `x_t` | `[B, 1]` |
| `t` | `[B]` |
| Time embedding | `[B, 64]` |
| Hidden layers | `[B, 128]` |
| Output $\hat{\varepsilon}$ | `[B, 1]` |

---

## Summary of Key Concepts

| Concept | Role |
|--------|------|
| Forward noising | Adds Gaussian noise via closed-form $q(x_t \mid x_0)$ |
| Closed-form diffusion sampling | Enables O(1) timestep sampling without recursion |
| Sinusoidal time conditioning | Injects temporal context into the network |
| Noise prediction parameterization | Predicts $\varepsilon$ rather than $x_0$ or score directly |
| Score learning in 1D | Recovers $\nabla_x \log p_t(x)$ via DSM on a tractable GMM |
