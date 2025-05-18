# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenSemanticScorer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        scores = self.fc(x).squeeze(-1)
        return torch.sigmoid(scores)

class MiniTransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        h = x
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x)
        x = h + attn_output
        h = x
        x = self.ln2(x)
        x = h + self.ff(x)
        return x

class SparseDepthTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, embed_dim))
        self.semantic_scorer = TokenSemanticScorer(embed_dim)
        self.layers = nn.ModuleList([MiniTransformerBlock(embed_dim) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        x = self.embed(x) + self.pos_embed[:, :T, :]
        semantic_scores = self.semantic_scorer(x)
        self.latest_layer_usage = torch.zeros_like(semantic_scores)

        for i, layer in enumerate(self.layers):
            threshold = i / len(self.layers)
            keep_mask = (semantic_scores > threshold)
            self.latest_layer_usage += keep_mask.float()

            x_new = x.detach().clone()  # avoid keeping gradients + free memory
            keep_indices = keep_mask.nonzero(as_tuple=True)

            if keep_indices[0].numel() > 0:
                selected_tokens = x[keep_indices].unsqueeze(0)
                selected_tokens_out = layer(selected_tokens).squeeze(0)
                x_new[keep_indices] = selected_tokens_out

            del x  # free the previous tensor
            x = x_new

        x = self.ln_final(x)
        return self.head(x)

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, embed_dim))
        self.layers = nn.ModuleList([MiniTransformerBlock(embed_dim) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        x = self.embed(x) + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.head(x)

# benchmark.py

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
