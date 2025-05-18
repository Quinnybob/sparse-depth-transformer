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

| Model     | Avg Layers/Token | Memory Saved | Runtime Change |
|-----------|------------------|---------------|----------------|
| Sparse    | ~3.5              | 5–15% ↓        | Slightly ↑     |
| Baseline  | 6.0               | –             | –              |

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
