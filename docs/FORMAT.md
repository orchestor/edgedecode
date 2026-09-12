# EdgeDecode v1 format

This format is original to this small reference implementation. It is **not** GGUF, AWQ, GPTQ, an Arm ABI, or a vendor NPU format.

For W[M,K], pad K to Kp=ceil(K/G)*G. Each row is split into groups of even size G. Let qmax=2^(b-1)-1 and s=max(abs(group))/qmax. A zero group uses s=1. Round W/s to nearest-even with NumPy `rint`, clip to [-qmax,qmax], and encode unsigned code=q+qmax. Four-bit code 15 and eight-bit code 255 are unused. Padding encodes zero.

Pack consecutive codes into low nibble first for 4-bit, one byte each for 8-bit. Scales are contiguous FP32 `[M,ceil(K/G)]`. The pilot uses groups 32,64,128.

Weight storage, including padding and scales but excluding host object metadata:

```
bytes = M*Kp*b/8 + 4*M*ceil(K/G)
```

Input activations, accumulation and outputs are FP32. The C kernel forms a code/activation dot product per group and then multiplies by its scale. It directly consumes packed bytes; it does not pre-expand the complete matrix to float. The F32 baseline uses the same simple row-dot structure, not BLAS. Neither backend is an optimized production baseline.

The Metal kernel assigns one output element to one thread and is likewise a correctness-oriented prototype. Its execution still needs validation on an accessible device. Packing, compilation and buffer creation are excluded from operator timing; composition timing includes allocation, dispatch, copies and tanh. These are intentionally distinct scopes.
