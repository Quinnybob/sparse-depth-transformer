import torch
import time
import pandas as pd
from model import SparseDepthTransformer, BaselineTransformer

vocab_size = 5000
embed_dim = 64
num_layers = 6
use_cuda = torch.cuda.is_available()
batch_sizes = [2, 8, 16]
seq_lengths = [20, 64, 128, 256, 512, 1024]

results = []

def benchmark_model(model_class, name, tokens, use_cuda):
    model = model_class(vocab_size, embed_dim, num_layers)
    if use_cuda:
        model = model.cuda()
        tokens = tokens.cuda()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        start = time.time()
        output = model(tokens)
        end = time.time()

    mem = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else None
    layer_usage = getattr(model, "latest_layer_usage", None)
    avg_layers = layer_usage.mean().item() if layer_usage is not None else None

    return {
        "Model": name,
        "Batch Size": tokens.shape[0],
        "Sequence Length": tokens.shape[1],
        "Time (s)": round(end - start, 4),
        "Memory (MB)": round(mem, 2) if mem is not None else None,
        "Avg Layers/Token": round(avg_layers, 3) if avg_layers is not None else None
    }

for batch in batch_sizes:
    for seq in seq_lengths:
        tokens = torch.randint(0, vocab_size, (batch, seq))
        results.append(benchmark_model(SparseDepthTransformer, "Sparse", tokens, use_cuda))
        results.append(benchmark_model(BaselineTransformer, "Baseline", tokens, use_cuda))

# Display results
pd.set_option('display.max_rows', None)
df = pd.DataFrame(results)
print(df.to_string(index=False))
