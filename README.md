# sequence-cleaning

🧬 A simple tool to clean and consolidate FASTA files with amino acid sequences.

## Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sequence-cleaning/blob/main/fasta_cleaner.ipynb)

1. Click the "Open in Colab" badge above
2. Run all cells (Runtime → Run all)
3. Upload your FASTA files using the upload button
4. Click "Process Files"
5. Download your cleaned CSV files

### Option 2: Local Jupyter
```bash
git clone https://github.com/YOUR_USERNAME/sequence-cleaning.git
cd sequence-cleaning
jupyter notebook fasta_cleaner.ipynb
```

## Features

- **Point-and-click interface** - No coding required
- **Multiple file support** - Upload and process multiple FASTA files at once
- **Automatic sequence cleaning** - Non-canonical amino acids replaced with `_`
- **Metadata parsing** - Extracts name, date, and other fields from headers
- **Duplicate handling** - Identical sequences are deduplicated; identical metadata with different sequences get version markers

## Output Files

### `sequences.csv`
| Column | Description |
|--------|-------------|
| `sequence_id` | Unique 12-character hash ID |
| `sequence` | Cleaned amino acid sequence |
| `length` | Sequence length |

### `metadata.csv`
| Column | Description |
|--------|-------------|
| `sequence_id` | Links to sequences.csv |
| `original_header` | Original FASTA header |
| `name` | Extracted protein name |
| `date` | Extracted date (if present) |
| `source_file` | Original filename |
| `field_N` | Additional parsed fields |

## Sample Data

A sample FASTA file is included at `sample_data/test_sequences.fasta` for testing.

## License

MIT License - see [LICENSE](LICENSE) for details.