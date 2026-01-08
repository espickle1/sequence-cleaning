"""
FASTA File Cleaner - Python Library

A library to clean and consolidate FASTA files with amino acid sequences.

Usage:
    from fasta_cleaner import process_fasta_files, clean_sequence, parse_header
    
    # Process files and get DataFrames
    sequences_df, metadata_df = process_fasta_files(["file1.fasta", "file2.fasta"])
    
    # Or process content directly
    sequences_df, metadata_df = process_fasta_content(fasta_string)
    
    # Save to CSV
    sequences_df.to_csv("sequences.csv", index=False)
    metadata_df.to_csv("metadata.csv", index=False)
"""

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# Canonical 20 amino acids
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Known database prefixes to skip when finding protein name
DB_PREFIXES = {"sp", "tr", "gb", "ref", "emb", "dbj", "pir", "prf", "uniprot"}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def clean_sequence(sequence: str) -> str:
    """
    Clean an amino acid sequence.
    
    - Convert to uppercase
    - Replace non-canonical amino acids with underscore
    - Remove whitespace, newlines, and non-letter characters
    
    Args:
        sequence: Raw amino acid sequence string
        
    Returns:
        Cleaned sequence with only canonical AAs and underscores
        
    Example:
        >>> clean_sequence("MKTXYZ")
        'MKT___'
    """
    sequence = sequence.upper().replace(" ", "").replace("\n", "").replace("\r", "")
    cleaned = ""
    for char in sequence:
        if char in CANONICAL_AA:
            cleaned += char
        elif char.isalpha():  # Non-canonical amino acid letter
            cleaned += "_"
        # Skip non-letter characters (numbers, symbols, etc.)
    return cleaned


def hash_sequence(sequence: str) -> str:
    """
    Generate a unique ID for a sequence using SHA-256.
    
    Args:
        sequence: Cleaned amino acid sequence
        
    Returns:
        First 12 characters of the SHA-256 hash
        
    Example:
        >>> hash_sequence("MKTAYI")
        'a1b2c3d4e5f6'
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:12]


def parse_header(header: str) -> Dict[str, str]:
    """
    Parse FASTA header to extract metadata fields.
    
    Handles common formats:
    - UniProt: sp|P12345|PROTEIN_NAME|date|...
    - GenBank: gb|ABC123|NAME|...
    - Custom: delimited by | ; / or tab
    
    Args:
        header: FASTA header line (with or without leading >)
        
    Returns:
        Dictionary with keys: original_header, name, date, field_1, field_2, ...
        Missing fields are empty strings (not filled with placeholders)
        
    Example:
        >>> parse_header("sp|P12345|HUMAN_INSULIN|2024-01-15")
        {'original_header': 'sp|P12345|HUMAN_INSULIN|2024-01-15',
         'name': 'HUMAN_INSULIN', 'date': '2024-01-15', 'field_1': 'P12345'}
    """
    # Remove leading > if present
    header = header.lstrip(">")
    
    result = {
        "original_header": header,
        "name": "",
        "date": ""
    }
    
    # Date pattern: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY, etc.
    date_pattern = re.compile(
        r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
    )
    
    # Try splitting by common delimiters
    delimiters = ["|", ";", "/", "\t"]
    fields = [header]
    
    for delim in delimiters:
        if delim in header:
            fields = [f.strip() for f in header.split(delim)]
            break
    
    # Process fields to find name and date
    extra_fields = []
    name_found = False
    
    for field in fields:
        field = field.strip()
        if not field:
            continue
        
        # Skip database prefixes
        if field.lower() in DB_PREFIXES:
            continue
        
        # Skip accession numbers (mostly alphanumeric, short)
        if re.match(r"^[A-Z0-9]{4,12}$", field) and not name_found:
            extra_fields.append(field)  # Keep as extra field
            continue
        
        # Check for date
        date_match = date_pattern.search(field)
        if date_match and not result["date"]:
            result["date"] = date_match.group(1)
            # If field is just the date, don't add to extras
            if field == date_match.group(1):
                continue
        
        # First meaningful field is the name
        if not name_found:
            result["name"] = field
            name_found = True
        else:
            extra_fields.append(field)
    
    # Add extra fields with numbered keys
    for i, field in enumerate(extra_fields, 1):
        result[f"field_{i}"] = field
    
    return result


def parse_fasta(content: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA format content.
    
    Args:
        content: String containing FASTA formatted sequences
        
    Returns:
        List of (header, sequence) tuples
        
    Example:
        >>> parse_fasta(">seq1\\nMKTAYI\\n>seq2\\nACDEFG")
        [('seq1', 'MKTAYI'), ('seq2', 'ACDEFG')]
    """
    sequences = []
    current_header = None
    current_seq = []
    
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        if line.startswith(">"):
            # Save previous sequence if exists
            if current_header is not None:
                sequences.append((current_header, "".join(current_seq)))
            current_header = line[1:]  # Remove >
            current_seq = []
        else:
            current_seq.append(line)
    
    # Don't forget the last sequence
    if current_header is not None:
        sequences.append((current_header, "".join(current_seq)))
    
    return sequences


def handle_duplicate_metadata(metadata_list: List[Dict]) -> List[Dict]:
    """
    Handle entries with identical metadata but distinct sequences.
    
    Appends version marker to name field (e.g., _v2, _v3) when the same
    name+date combination appears with different sequence IDs.
    
    Args:
        metadata_list: List of metadata dictionaries with sequence_id field
        
    Returns:
        Modified metadata list with version markers added where needed
    """
    # Group by (name, date) to find duplicates
    seen = defaultdict(list)
    
    for i, meta in enumerate(metadata_list):
        key = (meta.get("name", ""), meta.get("date", ""))
        seen[key].append(i)
    
    # Mark duplicates
    for key, indices in seen.items():
        if len(indices) > 1:
            # Check if sequences are actually different
            seq_ids = [metadata_list[i]["sequence_id"] for i in indices]
            if len(set(seq_ids)) > 1:  # Different sequences
                for version, idx in enumerate(indices, 1):
                    if version > 1:  # Don't mark the first one
                        original_name = metadata_list[idx].get("name", "")
                        metadata_list[idx]["name"] = f"{original_name}_v{version}"
    
    return metadata_list


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def process_fasta_content(
    content: str,
    source_name: str = "input"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process FASTA content string and return DataFrames.
    
    Args:
        content: String containing FASTA formatted sequences
        source_name: Name to use for source_file column in metadata
        
    Returns:
        Tuple of (sequences_df, metadata_df)
        
    Example:
        >>> seq_df, meta_df = process_fasta_content(fasta_string)
    """
    all_sequences = []
    all_metadata = []
    seen_sequences = {}
    
    parsed = parse_fasta(content)
    
    for header, raw_seq in parsed:
        # Clean sequence
        cleaned = clean_sequence(raw_seq)
        
        if not cleaned:
            continue
        
        # Get or create sequence ID
        if cleaned in seen_sequences:
            seq_id = seen_sequences[cleaned]
        else:
            seq_id = hash_sequence(cleaned)
            seen_sequences[cleaned] = seq_id
            all_sequences.append({
                "sequence_id": seq_id,
                "sequence": cleaned,
                "length": len(cleaned)
            })
        
        # Parse metadata
        meta = parse_header(header)
        meta["sequence_id"] = seq_id
        meta["source_file"] = source_name
        all_metadata.append(meta)
    
    # Handle duplicate metadata with different sequences
    all_metadata = handle_duplicate_metadata(all_metadata)
    
    # Create DataFrames
    sequences_df = pd.DataFrame(all_sequences)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Reorder metadata columns if not empty
    if not metadata_df.empty:
        priority_cols = ["sequence_id", "original_header", "name", "date", "source_file"]
        other_cols = [c for c in metadata_df.columns if c not in priority_cols]
        metadata_df = metadata_df[priority_cols + other_cols]
    
    return sequences_df, metadata_df


def process_fasta_files(
    file_paths: Union[str, Path, List[Union[str, Path]]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process one or more FASTA files and return DataFrames.
    
    Args:
        file_paths: Single file path or list of file paths
        
    Returns:
        Tuple of (sequences_df, metadata_df)
        
    Example:
        >>> seq_df, meta_df = process_fasta_files("proteins.fasta")
        >>> seq_df, meta_df = process_fasta_files(["file1.fa", "file2.fa"])
    """
    # Normalize to list
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    
    all_sequences = []
    all_metadata = []
    seen_sequences = {}
    
    for file_path in file_paths:
        file_path = Path(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        parsed = parse_fasta(content)
        
        for header, raw_seq in parsed:
            cleaned = clean_sequence(raw_seq)
            
            if not cleaned:
                continue
            
            if cleaned in seen_sequences:
                seq_id = seen_sequences[cleaned]
            else:
                seq_id = hash_sequence(cleaned)
                seen_sequences[cleaned] = seq_id
                all_sequences.append({
                    "sequence_id": seq_id,
                    "sequence": cleaned,
                    "length": len(cleaned)
                })
            
            meta = parse_header(header)
            meta["sequence_id"] = seq_id
            meta["source_file"] = file_path.name
            all_metadata.append(meta)
    
    all_metadata = handle_duplicate_metadata(all_metadata)
    
    sequences_df = pd.DataFrame(all_sequences)
    metadata_df = pd.DataFrame(all_metadata)
    
    if not metadata_df.empty:
        priority_cols = ["sequence_id", "original_header", "name", "date", "source_file"]
        other_cols = [c for c in metadata_df.columns if c not in priority_cols]
        metadata_df = metadata_df[priority_cols + other_cols]
    
    return sequences_df, metadata_df


def save_results(
    sequences_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Union[str, Path] = ".",
    prefix: str = ""
) -> Tuple[Path, Path]:
    """
    Save sequences and metadata DataFrames to CSV files.
    
    Args:
        sequences_df: DataFrame with sequence data
        metadata_df: DataFrame with metadata
        output_dir: Directory to save files (default: current directory)
        prefix: Optional prefix for filenames (e.g., "project1_")
        
    Returns:
        Tuple of (sequences_path, metadata_path)
        
    Example:
        >>> save_results(seq_df, meta_df, output_dir="output", prefix="batch1_")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seq_path = output_dir / f"{prefix}sequences.csv"
    meta_path = output_dir / f"{prefix}metadata.csv"
    
    sequences_df.to_csv(seq_path, index=False)
    metadata_df.to_csv(meta_path, index=False)
    
    return seq_path, meta_path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fasta_cleaner.py <file1.fasta> [file2.fasta ...]")
        print("\nProcesses FASTA files and outputs sequences.csv and metadata.csv")
        sys.exit(1)
    
    input_files = sys.argv[1:]
    print(f"Processing {len(input_files)} file(s)...")
    
    sequences_df, metadata_df = process_fasta_files(input_files)
    
    print(f"  Found {len(sequences_df)} unique sequences")
    print(f"  Found {len(metadata_df)} metadata entries")
    
    seq_path, meta_path = save_results(sequences_df, metadata_df)
    
    print(f"\nSaved to:")
    print(f"  {seq_path}")
    print(f"  {meta_path}")
