# Embedding Package
# Provides functions for FASTA cleaning and ESM-C embedding generation

from .fasta_cleaner import (
    clean_sequence,
    parse_header,
    parse_fasta,
    process_fasta_files,
    process_fasta_content,
    save_results,
)

from .esmc_embed_lib import (
    load_esmc_model,
    embed_sequences,
    embed_from_csv,
    embed_single,
    save_embeddings,
    load_embeddings,
)

__all__ = [
    # FASTA cleaning
    "clean_sequence",
    "parse_header", 
    "parse_fasta",
    "process_fasta_files",
    "process_fasta_content",
    "save_results",
    # Embedding
    "load_esmc_model",
    "embed_sequences",
    "embed_from_csv",
    "embed_single",
    "save_embeddings",
    "load_embeddings",
]
