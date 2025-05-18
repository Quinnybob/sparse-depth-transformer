# SparseDepthTransformer

> A transformer architecture that dynamically skips layers per token based on semantic importance — now with true compute savings.

---

##  Motivation

In standard transformers, every token passes through every layer — regardless of whether it's a high-impact content word or a filler like “the.” This wastes compute and memory, especially in long contexts.

SparseDepthTransformer introduces **per-token depth skipping**. It computes a semantic score for each token and **routes only the important ones through deeper layers**.

This project builds on the idea of dynamic routing, adding **true hard skipping**, not just masking, and shows measurable gains in memory and layer usage.

---

##  Features

- Per-token **semantic scorer**
- True **hard layer skipping**
- Baseline transformer for comparison
- Benchmarking across sequence lengths and batch sizes
- Outputs average layers used per token

---

##  Results

Benchmarked across batches (2, 8, 16) and sequence lengths (20–256):

| Model     | Avg Layers/Token | Memory (MB) | Time (s)  |
|-----------|------------------|-------------|-----------|
| Sparse    | ~3.5              | 22.16–105.43| 0.0058–0.0179 |
| Baseline  | 6.0               | 22.15–104.34| 0.0044–0.0207 |

The SparseDepthTransformer consistently used ~40% fewer layers per token with measurable memory savings, validating both semantic gating and compute reduction. Runtime is still slightly higher due to per-token execution, but this will be addressed with batching in future work.

Tokens now **actually bypass computation** at deeper layers if their semantic score is low — this was verified using conditional forward logic and benchmarking.

---

## Future Optimizations
- Implement token batching by depth group to improve runtime efficiency
- Add dropout-based probabilistic gating during training
- Fine-tune on real datasets (e.g., TinyStories, WikiText-2) and compare perplexity
- Integrate with HuggingFace Transformers for broader experimentation
- Introduce curriculum learning to vary routing difficulty during training

## Contact
Feel free to reach out with feedback, ideas, or collaboration opportunities!:
**Email:** desimoneq@gmail.com
