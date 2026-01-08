# Protein Sequence Analysis Pipeline

🧬 Complete pipeline for protein sequence cleaning, embedding, and analysis using ESM-C models.

## Pipeline Overview

```
FASTA → fasta_cleaner → CSV → esmc_embed → embeddings.pt → analysis → results
```

| Step | Tool | Description |
|------|------|-------------|
| 1 | `fasta_cleaner` | Clean sequences, parse metadata |
| 2 | `esmc_embed_lib` | Generate ESM-C embeddings |
| 3 | `entropy_lib` | Shannon entropy analysis |
| 4 | `logits_lib` | Amino acid propensity analysis |

---

## Quick Start

### Master Pipeline Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sequence-cleaning/blob/main/pipeline.ipynb)

The `pipeline.ipynb` notebook orchestrates the complete workflow with interactive widgets.

### Python Library Usage
```python
# Import from organized packages
from embedding import process_fasta_files, load_esmc_model, embed_from_csv
from analysis import analyze_entropy, analyze_residues, plot_heatmap

# Step 1: Clean FASTA
seq_df, meta_df = process_fasta_files("proteins.fasta")

# Step 2: Generate embeddings
model = load_esmc_model("hf_token")
results = embed_from_csv(model, "sequences.csv")

# Step 3: Entropy analysis
entropy = analyze_entropy(results)
print(f"Mean entropy: {entropy['global_mean']:.3f}")

# Step 4: Logits analysis
logits = analyze_residues(results, residues_of_interest={100: "D100"})
plot_heatmap(logits["scaled_logits"], logits["residue_labels"])
```

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

## Individual Tools

### FASTA Cleaning
```bash
python fasta_cleaner.py proteins.fasta
```

### Embedding Generation
```bash
python esmc_local_batch_embed.py config.yaml
```

---

## Output Files

| File | Description |
|------|-------------|
| `sequences.csv` | Cleaned sequences with unique IDs |
| `metadata.csv` | Parsed headers linked by sequence_id |
| `embeddings.pt` | PyTorch file with embeddings, logits, hidden states |
| `entropy_summary.csv` | Per-sequence entropy statistics |
| `logits_analysis.csv` | Amino acid propensities at positions of interest |

### Loading Results
```python
import torch

# Load embeddings
results = torch.load("embeddings.pt")
embedding = results["embeddings"][0]    # First sequence
mean_emb = embedding.mean(dim=0)        # Mean pooling

# Access hidden states (if extracted)
hidden = results["hidden_states"][0]    # First sequence
layer_12 = hidden.get(12)               # Layer 12
```

---

## Requirements

- Python 3.8+
- Core: `pandas`, `torch`
- Embedding: `esm`, `huggingface_hub`
- Analysis: `scikit-learn`, `matplotlib`

## License

MIT License - see [LICENSE](LICENSE)