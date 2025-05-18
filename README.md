This a working project on a novel transformer architecture that routes each token through a variable number of layers based on semantic importance, reducing unnecessary computation and memory usage.


Modern transformer models waste compute by sending **every token** through **every layer**, regardless of how much semantic content each token actually carries.

But do common tokens like `"the"` or `"and"` really need the same deep processing as `"DNA"` or `"restructure"`?

**SparseDepthTransformer** introduces a token-wise routing mechanism that decides — in real-time — how many layers a token should pass through, based on its **semantic density**. This enables **layer skipping** on a per-token basis, which is rarely explored in transformer research.

---

## Goal

- Score each token’s semantic importance using a learned linear probe
- Route important tokens through all layers
- Let less important tokens **skip deeper layers**
- Use **hard skipping** (not soft blending) to save real compute

This leads to a transformer that is **both smarter and lighter**.

---

## 📊 Benchmark Results

Benchmarked on 10 runs, using sequence length = 20, batch size = 2:

| Model | Time (sec) | Max Memory (MB) | Avg Layers per Token |
|-------|------------|------------------|-----------------------|
| **SparseDepthTransformer** | ~0.0049 | ~23.1 | ~3.6 |
| **Baseline Transformer**   | ~0.0037 | ~27.0 | 6.0 |

**~40% fewer layers processed per token**  
**~15% less GPU memory used**  
⚠Slight increase in latency due to token-level execution — batching optimization planned

---

## Features

- ✅ Semantic scorer module
- ✅ Hard-skipping per-token per-layer
- ✅ Layer usage tracking
- ✅ Baseline transformer for comparison
- ✅ Benchmarking script for time, memory, and depth

---

## 📁 Run It Yourself

Install dependencies:
```bash
pip install torch
