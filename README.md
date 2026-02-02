# Protein Sequence Analysis Pipeline

Complete pipeline for protein sequence cleaning, ESM-C embedding, entropy analysis, and ChimeraX visualization. Each step processes sequences individually through per-sequence functions.

## Repository Structure

```
sequence-cleaning/
├── pipeline.ipynb                 # Main orchestration notebook (recommended entry point)
├── chimerax_color_demo.ipynb      # ChimeraX visualization script generator
├── fasta_cleaner.ipynb            # Interactive FASTA cleaning notebook
├── esmc_embed_colab.ipynb         # Interactive embedding generator for Colab
│
├── embedding/                     # Embedding package
│   ├── __init__.py
│   ├── fasta_cleaner.py           # FASTA parsing and cleaning functions
│   └── esmc_embed_lib.py          # ESM-C model loading and embedding
│
├── analysis/                      # Analysis package
│   ├── __init__.py
│   ├── entropy_lib.py             # Shannon entropy calculation
│   ├── logits_lib.py              # Logits pooling and heatmap visualization
│   └── chimerax_color_lib.py      # ChimeraX .cxc script generation
│
├── fasta_cleaner.py               # Standalone FASTA cleaner script
├── esmc_embed_lib.py              # Standalone embedding library
├── esmc_local_batch_embed.py      # YAML config-based batch embedding
├── shannon_entropy_calculation_gpu.py  # GPU-optimized entropy calculation
├── esmc_logits_pooling_and_visualization.py  # Logits analysis script
│
├── sample_data/                   # Example files
│   ├── test_sequences.fasta       # Sample FASTA with various formats
│   └── entropy_color_mapping_example.cxc  # Example ChimeraX output
│
├── environment.yaml               # Conda environment specification
├── ruff.toml                      # Code linting configuration
└── LICENSE                        # MIT License
```

## Pipeline Overview

```
FASTA → Clean → Embed → Analyze → Visualize
  │        │       │        │          │
  │        │       │        │          └─► ChimeraX .cxc scripts
  │        │       │        │
  │        │       │        ├─► Entropy analysis (per-residue conservation)
  │        │       │        └─► Logits analysis (amino acid propensities)
  │        │       │
  │        │       └─► ESM-C embeddings + logits (PyTorch tensor)
  │        │
  │        └─► sequences.csv + metadata.csv
  │
  └─► Raw protein sequences
```

| Step | Module | Key Function | Description |
|------|--------|--------------|-------------|
| 1 | `fasta_cleaner.py` | `process_fasta_content()` | Parse FASTA, clean sequences, extract metadata |
| 2 | `esmc_embed_lib.py` | `embed_single()` | Generate ESM-C embeddings for one sequence |
| 3 | `entropy_lib.py` | `calculate_entropy()` | Shannon entropy per residue |
| 4 | `logits_lib.py` | `analyze_residues()` | Amino acid propensity analysis |
| 5 | `chimerax_color_lib.py` | `generate_chimerax_script()` | Create ChimeraX visualization scripts |

---

## Quick Start

### Option A: Pipeline Notebook (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espickle1/sequence-cleaning/blob/main/pipeline.ipynb)

1. Open `pipeline.ipynb` in Jupyter or Google Colab
2. Run the **Setup** cell to install dependencies
3. **Step 1** — Upload `.fasta` files using the upload widget
4. **Step 2** — Enter HuggingFace token, select model (`esmc_300m` or `esmc_600m`), generate embeddings
5. **Step 3** — Run entropy analysis (per-residue conservation scoring)
6. **Step 4** — Run logits analysis (amino acid propensity heatmaps)
7. **Step 5** — Export results to the `results/` folder

### Option B: ChimeraX Visualization Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/espickle1/sequence-cleaning/blob/main/chimerax_color_demo.ipynb)

1. Open `chimerax_color_demo.ipynb` in Google Colab
2. Upload your CSV files (metadata, sequences, entropy)
3. Configure color mapping options (colormap, transform, model/chain IDs)
4. Optionally apply value cutoffs to filter residues
5. Download generated `.cxc` scripts for ChimeraX

---

## Step-by-Step Instructions

### Step 1: Clean FASTA Files

The FASTA cleaner parses protein sequences and extracts metadata from headers.

**Supported header formats:**
- UniProt: `>sp|P12345|PROTEIN_NAME|DATE|TISSUE`
- GenBank: `>gb|ABC123|NAME`
- Custom: `>custom_format;ProteinX;2023-11-30;Lab_Sample`

**Using the notebook:**
```python
from embedding.fasta_cleaner import process_fasta_files

seq_df, meta_df = process_fasta_files("proteins.fasta")
# seq_df: sequence_id, sequence, length
# meta_df: sequence_id, original_header, name, date, source_file, field_1, ...
```

**Command line:**
```bash
python fasta_cleaner.py input.fasta -o results/
```

**What it does:**
- Converts sequences to uppercase
- Replaces non-canonical amino acids (X, B, J, Z) with underscores
- Generates unique SHA-256 based sequence IDs (first 12 characters)
- Parses metadata fields from headers
- Handles duplicate sequences with version markers

### Step 2: Generate Embeddings

Load an ESM-C model and embed sequences to get per-residue representations.

```python
from embedding.esmc_embed_lib import load_esmc_model, embed_single

# Load model (requires HuggingFace token)
model = load_esmc_model("your_hf_token", model_name="esmc_300m")

# Embed a single sequence
result = embed_single(
    model,
    sequence="MKTAYIAK...",
    return_embeddings=True,
    return_logits=True
)

# result["embeddings"]: shape (seq_len, 1536)
# result["logits"]: shape (seq_len, 64)
```

**Hidden layer extraction:**
```python
# Single layer
result = embed_single(model, seq, hidden_layers=12)

# Multiple layers
result = embed_single(model, seq, hidden_layers=[12, 24, 36])

# All layers
result = embed_single(model, seq, hidden_layers="all")
```

**Batch embedding from CSV:**
```python
from embedding.esmc_embed_lib import embed_from_csv

results = embed_from_csv(
    model,
    "sequences.csv",
    sequence_col="sequence",
    id_col="sequence_id",
    return_embeddings=True,
    return_logits=True
)
```

### Step 3: Entropy Analysis

Calculate Shannon entropy to identify conserved and flexible positions.

```python
from analysis.entropy_lib import calculate_entropy, analyze_entropy

# Per-residue entropy from logits tensor
entropy_values = calculate_entropy(logits, base="e")  # or "2", "10"

# Full analysis with position classification
results = analyze_entropy(
    embeddings_dict,  # {"sequence_id": [...], "logits": [...]}
    base="e",
    constrained_percentile=10.0,  # Bottom 10% = constrained
    flexible_percentile=90.0      # Top 10% = flexible
)
```

**Output includes:**
- Per-residue entropy values
- Position classifications (constrained, intermediate, flexible)
- Summary statistics (mean, std, min, max entropy)

### Step 4: Logits Analysis

Analyze amino acid propensities at each residue position.

```python
from analysis.logits_lib import analyze_residues, plot_heatmap, AA_VOCAB

# Analyze all positions in a sequence
results = analyze_residues(
    embeddings_dict,
    residues_of_interest={i: f"Pos {i+1}" for i in range(seq_length)},
    pool_method="mean",      # mean, max, sum
    scale_method="softmax"   # minmax, robust, power, standard, softmax
)

# Generate heatmap visualization
plot_heatmap(
    results["probs"],
    results["residue_labels"],
    AA_VOCAB  # 20 canonical amino acids
)
```

### Step 5: ChimeraX Visualization

Generate `.cxc` scripts to visualize entropy or other per-residue values in ChimeraX.

```python
from analysis.chimerax_color_lib import (
    generate_chimerax_script,
    write_chimerax_script,
    scale_values,
    fit_scaler,
    fit_minmax_scaler,
    parse_range_string,
    create_value_mask
)

# Basic usage - single file
script = generate_chimerax_script(
    entropy_values,           # 1D array of per-residue values
    cmap_name="RdBu",         # Colormap name
    transform_method="quantile",  # none, log, minmax, quantile, power, standard, robust
    color=True,
    color_invert=True,        # Invert colormap direction
    transparency=False,
    model=1,                  # ChimeraX model number
    chain="A"                 # ChimeraX chain ID
)

write_chimerax_script(script, "entropy_colors.cxc")
```

**Multi-file normalization (consistent colors across proteins):**
```python
# Combine values from multiple proteins
combined = np.concatenate([values1, values2, values3])

# Fit scalers on combined data
transform_scaler = fit_scaler(combined, method="quantile")
scaled = scale_values_with_scaler(combined, transform_scaler)
minmax_scaler = fit_minmax_scaler(scaled)

# Generate scripts with shared scale
from analysis.chimerax_color_lib import generate_chimerax_script_with_scalers

script = generate_chimerax_script_with_scalers(
    values1,
    transform_scaler=transform_scaler,
    minmax_scaler=minmax_scaler,
    cmap_name="Greys",
    model=1,
    chain="A"
)
```

**Value cutoffs (filter residues by value ranges):**
```python
# Parse range specification
ranges = parse_range_string("0.9-1.0")  # Top 10% for quantile transform

# Create boolean mask
mask = create_value_mask(transformed_values, ranges)

# Generate script with filtered residues
script = generate_chimerax_script(
    values,
    transform_method="quantile",
    residue_mask=mask,  # Only color residues where mask is True
    cmap_name="RdBu"
)
```

**Using the notebook:**

The `chimerax_color_demo.ipynb` notebook provides an interactive workflow:

1. **Upload files**: metadata.csv, sequences.csv, entropy.csv
2. **Multi-file mode** (optional): Upload multiple entropy files for consistent normalization
3. **Configure**: colormap, transform method, color/transparency options
4. **Value cutoffs** (optional): Filter by transformed value ranges (e.g., top 10% entropy)
5. **Generate**: Creates one `.cxc` file per sequence with model/chain specifications
6. **Download**: Files ready to open in ChimeraX

---

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `sequences.csv` | CSV | Cleaned sequences with unique IDs |
| `metadata.csv` | CSV | Parsed headers linked by sequence_id |
| `embeddings.pt` | PyTorch | Embeddings, logits, and hidden states per sequence |
| `entropy_summary.csv` | CSV | Per-sequence entropy statistics |
| `entropy_per_residue_*.csv` | CSV | Per-residue entropy with classifications |
| `logits_analysis.csv` | CSV | Softmax amino acid probabilities |
| `*.cxc` | ChimeraX | Per-residue color/transparency commands |

### Loading PyTorch Results

```python
import torch

# Load embeddings file
data = torch.load("embeddings.pt")

# Access by sequence
seq_id = data["sequence_id"][0]
embedding = data["embeddings"][0]     # Shape: (seq_len, 1536)
logits = data["logits"][0]            # Shape: (seq_len, 64)

# Mean pooling for sequence-level embedding
mean_emb = embedding.mean(dim=0)      # Shape: (1536,)
```

---

## Transform Methods

The pipeline supports multiple scaling/transformation methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| `none` | No transformation | Raw values |
| `log` | Log(1+x) transform | Compress large value ranges |
| `minmax` | Scale to [0, 1] | Normalize within a single file |
| `quantile` | Rank-based (0-1) | Ordinal ranking, outlier-resistant |
| `power` | Yeo-Johnson transform | Handle skewed distributions |
| `standard` | Z-score normalization | Mean=0, std=1 |
| `robust` | Median/IQR scaling | Outlier-resistant |

---

## Installation

### Using Conda (Recommended)

```bash
git clone https://github.com/espickle1/sequence-cleaning.git
cd sequence-cleaning
conda env create -f environment.yaml
conda activate sequence-cleaning
pip install esm huggingface_hub
```

### Using pip

```bash
pip install pandas numpy torch scikit-learn seaborn matplotlib jupyter
pip install esm huggingface_hub
```

### Requirements

- Python 3.9+
- Core: `pandas`, `numpy`, `torch`
- Embedding: `esm`, `huggingface_hub` (requires HuggingFace account with token)
- Analysis: `scikit-learn`, `seaborn`, `matplotlib`
- Optional: CUDA for GPU acceleration
- Visualization: [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/) (for viewing `.cxc` files)

---

## Python Library Usage

```python
from embedding.fasta_cleaner import process_fasta_files
from embedding.esmc_embed_lib import load_esmc_model, embed_single
from analysis.entropy_lib import calculate_entropy
from analysis.logits_lib import analyze_residues, plot_heatmap, AA_VOCAB
from analysis.chimerax_color_lib import generate_chimerax_script, write_chimerax_script

# Step 1: Clean FASTA
seq_df, meta_df = process_fasta_files("proteins.fasta")

# Step 2: Load model
model = load_esmc_model("hf_token")

# Steps 2-5: Process each sequence
for _, row in seq_df.iterrows():
    seq_id = row["sequence_id"]
    sequence = row["sequence"]

    # Embed
    emb = embed_single(model, sequence, return_embeddings=True, return_logits=True)

    # Entropy analysis
    entropy = calculate_entropy(emb["logits"], base="e")
    print(f"{seq_id}: mean entropy = {entropy.mean():.3f}")

    # Logits analysis
    residues = {i: f"Pos {i+1}" for i in range(len(sequence))}
    logits_result = analyze_residues(
        {"sequence_id": [seq_id], "logits": [emb["logits"]]},
        residues_of_interest=residues,
        pool_method="mean",
        scale_method="softmax"
    )
    plot_heatmap(logits_result["probs"], logits_result["residue_labels"], AA_VOCAB)

    # ChimeraX visualization
    script = generate_chimerax_script(
        entropy,
        cmap_name="RdBu",
        transform_method="quantile",
        color_invert=True,
        model=1,
        chain="A"
    )
    write_chimerax_script(script, f"{seq_id}_entropy.cxc")
```

---

## License

Apache License 2.0 - see [LICENSE](https://github.com/espickle1/sequence-cleaning/blob/main/LICENSE)
