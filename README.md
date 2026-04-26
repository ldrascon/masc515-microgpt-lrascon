# masc515-microgpt-lrascon

## Mixture of Experts (MoE) Branch
Here, we replace the dense feed-forward network (FFN) in the Transformer block with an MoE layer:
- multiple FFN “experts”
- a learned router (gate) that assigns each token to expert(s)

In a standard MoE formulation:
- experts are FFNs
- the router produces weights via a softmax gate
- output is a weighted combination of expert outputs

y = Σ_i G(x)_i * E_i(x)
with G(x) = Softmax(x W_g)

**Implementation**
- The original MLP block (fc1 → GELU → fc2) is replaced by MOE_NUM_EXPERTS experts, each an independent fc1/fc2 FFN.
- A router matrix `moe_gate` produces expert logits from the token representation.
- We use a dense (soft) mixture: compute all experts and combine with the router softmax weights.
