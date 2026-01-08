# Sequence Cleaning & Embedding Pipeline

🧬 Tools to clean FASTA files and generate protein sequence embeddings using ESM-C models.

## Pipeline Overview

```
FASTA Files → fasta_cleaner → CSV Files → esmc_embed → Embeddings (.pt)
```

1. **FASTA Cleaner**: Clean sequences, parse metadata, deduplicate
2. **ESMC Embed**: Generate embeddings using ESM-C models

---

## Step 1: FASTA Cleaning

### Option A: Google Colab (Recommended for beginners)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sequence-cleaning/blob/main/fasta_cleaner.ipynb)

1. Click the badge above
2. Run all cells (Runtime → Run all)
3. Upload your FASTA files
4. Click "Process Files"
5. Download `sequences.csv` and `metadata.csv`

### Option B: Python Library
```python
from fasta_cleaner import process_fasta_files, save_results

sequences_df, metadata_df = process_fasta_files(["proteins.fasta"])
save_results(sequences_df, metadata_df, output_dir="output")
```

### Option C: Command Line
```bash
python fasta_cleaner.py file1.fasta file2.fasta
```

---

## Step 2: Protein Embeddings

Generate embeddings from the cleaned `sequences.csv` file.

### Option A: Google Colab (Recommended for beginners)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sequence-cleaning/blob/main/esmc_embed_colab.ipynb)

1. Click the badge above
2. Enter your HuggingFace token
3. Upload `sequences.csv` from Step 1
4. Click "Generate Embeddings"
5. Download `embeddings.pt`

### Option B: Python Library
```python
from esmc_embed_lib import load_esmc_model, embed_from_csv, save_embeddings

model = load_esmc_model("your_hf_token")
results = embed_from_csv(model, "sequences.csv")
save_embeddings(results, "embeddings.pt")
```

### Option C: Config-Based Script
```bash
python esmc_local_batch_embed.py config.yaml
```

---

## Output Files

### From FASTA Cleaner

| File | Column | Description |
|------|--------|-------------|
| `sequences.csv` | `sequence_id` | Unique 12-character hash ID |
| | `sequence` | Cleaned amino acid sequence |
| | `length` | Sequence length |
| `metadata.csv` | `sequence_id` | Links to sequences.csv |
| | `original_header` | Original FASTA header |
| | `name` | Extracted protein name |
| | `date` | Extracted date (if present) |
| | `source_file` | Original filename |

### From ESMC Embed

| File | Key | Description |
|------|-----|-------------|
| `embeddings.pt` | `sequence_id` | Links to metadata.csv |
| | `embeddings` | Per-residue embedding tensors |
| | `logits` | Logits tensors |
| | `model_name` | ESM-C model used |

### Loading Embeddings
```python
import torch

results = torch.load("embeddings.pt")
embedding = results["embeddings"][0]  # First sequence
mean_emb = embedding.mean(dim=0)      # Mean pooling
```

---

## File Structure

```
sequence-cleaning/
├── fasta_cleaner.py          # Library: FASTA cleaning functions
├── fasta_cleaner.ipynb       # Notebook: Interactive FASTA cleaner
├── esmc_embed_lib.py         # Library: Embedding functions
├── esmc_embed_colab.ipynb    # Notebook: Interactive embedding generator
├── esmc_local_batch_embed.py # Script: Config-based batch embedding
├── sample_data/
│   └── test_sequences.fasta  # Sample FASTA file
└── README.md
```

## Requirements

- Python 3.8+
- pandas
- For embeddings: `esm`, `huggingface_hub`, `torch`

## License

MIT License - see [LICENSE](LICENSE) for details.