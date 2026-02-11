# Part 1 — PyTorch Foundations

## Goal
Be able to implement attention + backprop.

## Resources (do these)
- PyTorch tensors/autograd tutorial:
  https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html
- Learning PyTorch with Examples (training loops + nn.Module):
  https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html

## Scripts to test knowledge
1) `autograd_sanity.py`
- Create tensor with `requires_grad=True`
- Make scalar loss
- `backward()`
- Check `.grad` is non-None and finite

2) `trainloop_linear.py`
- Minimal training loop (optimizer, zero_grad, backward, step)
- Fit a tiny synthetic regression; confirm loss decreases

3) `attention_einsum.py`
- Implement scaled dot-product attention with `einsum` (batched + multi-head):
  - Q,K,V projections
  - reshape to `[B,H,N,Dh]`
  - `scores = einsum('bhid,bhjd->bhij', q, k) / sqrt(Dh)`
  - softmax + dropout
  - `out = einsum('bhij,bhjd->bhid', attn, v)`
  - merge heads + output proj

4) `transformer_block.py`
- Wrap attention in a block:
  - LayerNorm
  - residual connection
  - dropout
  - simple 2-layer MLP
