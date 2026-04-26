# masc515-microgpt-lrascon

## Gaussian Error Linear Unit (GELU) Branch
Here, we replace the the ReLU gating with a smooth, probablistic weighting.
The GELU activation, as sourced from Hendrycks and Gimpel, is defined as GELU(x) = x((phi)(x)), where phi is the standard normal CDF.
As suggested in the papter, the fast approximation was used: GELU(x) = ~ x*(sigma)(1.702x), where sigma is the sigmoid function.
