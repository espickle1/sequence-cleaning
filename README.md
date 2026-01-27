# Protein Sequence Analysis Pipeline

Complete pipeline for protein sequence cleaning, embedding, and analysis using ESM-C models. Each step processes **one sequence at a time** through per-sequence functions.

## Pipeline Overview

```
FASTA → fasta_cleaner → CSV → embed → entropy → logits → results
                               ↑         ↑         ↑
                          per-sequence functions at every step
```

| Step | Function | Description |
|------|----------|-------------|
| 1 | `process_fasta_content()` | Clean sequences, parse metadata |
| 2 | `run_embedding_for_sequence()` | Generate ESM-C embeddings for one sequence |
| 3 | `run_entropy_for_sequence()` | Shannon entropy for one sequence |
| 4 | `run_logits_for_sequence()` | Amino acid propensity (softmax) for one sequence |

---

## Quick Start

### Option A: Pipeline Notebook (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sequence-cleaning/blob/main/pipeline.ipynb)

1. Open `pipeline.ipynb` in Jupyter or Google Colab.
2. Run the **Setup** cell to install dependencies and load packages.
3. **Step 1** — Upload one or more `.fasta` files using the upload widget.
4. **Step 2** — Enter your HuggingFace token, choose a model, click **Load Model**, then **Generate Embeddings**. Each sequence is embedded individually via `run_embedding_for_sequence()`.
5. **Step 3** — Entropy analysis runs automatically per sequence. Each sequence gets its own entropy profile plot.
6. **Step 4** — Logits analysis runs per sequence across every residue position. Heatmaps show softmax probabilities for each sequence.
7. **Step 5** — Export all results to the `results/` folder.

### Option B: Python Library Usage

```python
from embedding.fasta_cleaner import process_fasta_files
from embedding.esmc_embed_lib import load_esmc_model, embed_single
from analysis.entropy_lib import analyze_entropy
from analysis.logits_lib import analyze_residues, plot_heatmap, AA_VOCAB

# Step 1: Clean FASTA
seq_df, meta_df = process_fasta_files("proteins.fasta")

# Step 2: Load model
model = load_esmc_model("hf_token")

# Steps 2–4: Process each sequence individually
for _, row in seq_df.iterrows():
    seq_id = row["sequence_id"]
    sequence = row["sequence"]

    # Embed one sequence
    emb = embed_single(model, sequence, return_embeddings=True, return_logits=True)

    # Entropy for one sequence
    entropy = analyze_entropy(
        {"sequence_id": [seq_id], "logits": [emb["logits"]]},
        base="e", constrained_percentile=10.0, flexible_percentile=90.0
    )
    print(f"{seq_id}: mean entropy = {entropy['mean_entropy'][0]:.3f}")

    # Logits for one sequence (all residue positions)
    seq_length = emb["logits"].shape[0]
    residues = {i: f"Position {i+1}" for i in range(seq_length)}
    logits = analyze_residues(
        {"sequence_id": [seq_id], "logits": [emb["logits"]]},
        residues_of_interest=residues,
        pool_method="mean", scale_method="minmax"
    )

    # Heatmap shows softmax probabilities
    plot_heatmap(logits["probs"], logits["residue_labels"], AA_VOCAB)
```

---

## Per-Sequence Functions

The pipeline notebook defines three functions that each operate on a single protein:

### `run_embedding_for_sequence(model, seq_id, sequence)`

Embeds one sequence using `embed_single`. Returns a dict with `seq_id`, `embeddings`, and `logits` tensors.

### `run_entropy_for_sequence(seq_id, logits)`

Wraps a single sequence into the format expected by `analyze_entropy` and returns per-residue entropy, constrained positions, and flexible positions.

### `run_logits_for_sequence(seq_id, logits)`

Builds `residues_of_interest` covering every position in the sequence, wraps the data for `analyze_residues`, and returns softmax probabilities for all 20 standard amino acids at each position.

---

## File Structure

```
sequence-cleaning/
├── pipeline.ipynb              # Master orchestration notebook
├── embedding/                  # Embedding package
│   ├── __init__.py
│   ├── fasta_cleaner.py        # FASTA cleaning functions
│   └── esmc_embed_lib.py       # ESM-C embedding functions
├── analysis/                   # Analysis package
│   ├── __init__.py
│   ├── entropy_lib.py          # Shannon entropy analysis
│   └── logits_lib.py           # Logits pooling & visualization
├── fasta_cleaner.ipynb         # Interactive FASTA cleaner
├── esmc_embed_colab.ipynb      # Interactive embedding generator
└── esmc_local_batch_embed.py   # Config-based batch embedding
```

---

## Output Files

| File | Description |
|------|-------------|
| `sequences.csv` | Cleaned sequences with unique IDs |
| `metadata.csv` | Parsed headers linked by sequence_id |
| `embeddings.pt` | PyTorch file with embeddings and logits per sequence |
| `entropy_summary.csv` | Per-sequence entropy statistics |
| `logits_analysis.csv` | Softmax amino acid probabilities at every position |

### Loading Results

```python
import torch

# Load embeddings
results = torch.load("embeddings.pt")
embedding = results["embeddings"][0]    # First sequence
mean_emb = embedding.mean(dim=0)        # Mean pooling

# Access logits
logits = results["logits"][0]           # First sequence logits
```

---

## Requirements

- Python 3.8+
- Core: `pandas`, `torch`
- Embedding: `esm`, `huggingface_hub`
- Analysis: `scikit-learn`, `matplotlib`

## License

MIT License - see [LICENSE](LICENSE)
