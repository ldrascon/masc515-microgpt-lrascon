# masc515-microgpt-lrascon

## Low-Rank Adaptation (LoRA) Branch
LoRA is a parameter-efficient fine-tuning method for large models.
Here, rather than updating a full-weight matrix during adaptation, LoRA freezes the pretrained weight and learns a low-rank update that is much smaller than the original matrix, per Hu et al.
The GELU activation, as sourced from Hendrycks and Gimpel, is defined as GELU(x) = x((phi)(x)), where phi is the standard normal CDF.
As suggested in the papter, the fast approximation was used: GELU(x) = ~ x*(sigma)(1.702x), where sigma is the sigmoid function.

For a dense layer with pretrained weights \(W_0 \in \mathbb{R}^{d \times k}\), LoRA reparameterizes the update as:

\[
W = W_0 + \Delta W,\quad \Delta W = BA
\]

where \(A \in \mathbb{R}^{r \times k}\), \(B \in \mathbb{R}^{d \times r}\), and the rank \(r \ll \min(d,k)\).  
For an input vector \(x\), the forward pass becomes:

\[
h = W_0 x + \frac{\alpha}{r} \, B(Ax)
\]

This cuts down the number of trainable parameters since only \(A\) and \(B\) are trained while \(W_0\) stays constant.

### Initialization
Following the LoRA paper’s recommended approach (Hu et al.), the low-rank update starts at zero so the model initially behaves like the baseline model, in which:

- \(A\) is initialized randomly (Gaussian)
- \(B\) is initialized to zeros, so \(BA = 0\) at step 0 :contentReference[oaicite:3]{index=3}

### Implementation
This microgpt implementation applies LoRA to the Transformer self-attention projection matrices:

- Query projection: `attn_wq`
- Value projection: `attn_wv`

(i.e., LoRA is used when constructing `q` and `v` inside the attention block).

### Implementation details in microgpt.py
- Added LoRA matrices to `state_dict`:
  - `layer{i}.attn_wq.lora_A`, `layer{i}.attn_wq.lora_B`
  - `layer{i}.attn_wv.lora_A`, `layer{i}.attn_wv.lora_B`
- Implemented LoRA-based linear transform:
  - `linear_lora(x, W, A, B, alpha)` computes `Wx + (alpha/r)*B(Ax)`
- Optional “true LoRA freezing”:
  - when enabled, `params` is constructed to include *only* `.lora_` parameters so that the base model weights do not receive Adam updates. 

### Hyperparameters used
- `LORA_R` (rank): controls the low-rank dimension \(r\)
- `LORA_ALPHA` (scaling): controls \(\alpha\)
- scale factor implemented as \(\alpha/r\) 

### Notes / limitations
LoRA is primarily motivated as an **adaptation method for pretrained models**. LoRA is included here to demonstrate the algorithmic mechanism and how low-rank updates can be pushed into attention layers. 
