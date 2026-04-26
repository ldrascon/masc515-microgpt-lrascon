# masc515-microgpt-lrascon

## Rotary Position Embedding (RoPE) Branch
RoPE encodes position by rotating the query and key vectors by a position-dependent rotation matrix instead of adding an absolute positional embedding. 
This injects absolute position information while making attention scores depend on relative position.

In RoFormer, RoPE is defined using a block-diagonal rotation matrix with 2×2 rotation blocks (Eq. 15 of Su et al.), parameterized by angles based on θ_i = 10000^{-2(i-1)/d}. Queries and keys are rotated as:
- q_m = R_{Θ,m} W_q x_m
- k_n = R_{Θ,n} W_k x_n

This rotation to an attention inner product that depends on the relative offset (Eq. 16 of Su et al.).

**Implementation:**
- Apply RoPE to Q and K after projection.
- Implement rotation on each attention head by rotating pairs of dimensions (0,1), (2,3), … using cos/sin computed from position.
- When RoPE is enabled, the learned absolute positional embedding `wpe` is disabled (set to zero) to avoid double-encoding position.

