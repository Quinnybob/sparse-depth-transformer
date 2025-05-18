# SparseDepthTransformer: Per-Token Dynamic Depth Routing for Efficient Transformers

> A novel transformer architecture that routes each token through a variable number of layers based on semantic importance — reducing memory usage and unnecessary compute.

---

##Motivation

Modern transformer models waste compute by sending **every token** through **every layer**, regardless of how much semantic content each token actually carries.

But do common tokens like "the" or "and" really need the same deep processing as "DNA" or "restructure"?

**SparseDepthTransformer** introduces a token-wise routing mechanism that decides — in real-time — how many layers a token should pass through, based on its **semantic density**. This enables **layer skipping** on a per-token basis, which is rarely explored in transformer research.

---

## Core Idea

- Score each token’s semantic importance using a learned linear probe
- Route important tokens through all layers
- Let less important tokens **skip deeper layers**
- Use **hard skipping** (not soft blending) to save real compute

This leads to a transformer that is **both smarter and lighter**.

---

##  Benchmark Results

Benchmarked on 10 runs, using sequence length = 20, batch size = 2:

| Model                    | Time (sec) | Max Memory (MB) | Avg Layers per Token |
|-------------------------|------------|------------------|-----------------------|
| **SparseDepthTransformer** | ~0.0049     | ~23.1            | ~3.6                  |
| **Baseline Transformer**   | ~0.0037     | ~27.0            | 6.0                  |

**~40% fewer layers processed per token**  
**~15% less GPU memory used**  
~Slight increase in latency due to token-level execution — batching optimization planned

---

##  Features

- Semantic scorer module
- Hard-skipping per-token per-layer
- Layer usage tracking
- Baseline transformer for comparison
- Benchmarking script for time, memory, and depth

---

##  How to Run

###  Installation
```bash
pip install torch
```

###  Running the Benchmark
```bash
python main.py
```

###  Run in Colab
Use this notebook: [Open Colab](https://colab.research.google.com/) *(https://colab.research.google.com/drive/1UDcoTnULE0fUJKJiJjsjMPei_qPVMV6p#scrollTo=Sx_UFQTYBkm9)*

---

##  Why This Project Matters

This work explores **depth sparsity**, a rarely studied axis in transformer optimization.  
While attention sparsity and Mixture-of-Experts (MoE) are popular, per-token **layer skipping** introduces a new degree of freedom — enabling models to **spend compute only where it's needed**.

> "Not every token deserves the same amount of thought."

This could enable:
- Lightweight models for mobile and edge AI
- Faster inference on long-context inputs
- Adaptive compute strategies in LLMs

---

## Future Work

- [ ] Batch tokens by routing level for GPU efficiency
- [ ] Train on real data (TinyStories, Alpaca, etc.)
- [ ] Add routing visualizations and attention heatmaps
- [ ] Integrate with HuggingFace Transformers for broader use
- [ ] Experiment with curriculum depth scheduling

---

## Author

This project was created by **Quinnybob**, a high school researcher focused on efficient AI systems and transformer interpretability.

📧 desimoneq@gmail.com
🔗 [GitHub Profile](https://github.com/Quinnybob)

---

## Contributing / Contact

Feel free to reach out if:
- You're a researcher or student interested in collaborating
- You want to integrate this into another model
- You’d like to contribute optimizations or training runs

Pull requests, ideas, and discussions are welcome!

MIT License.
