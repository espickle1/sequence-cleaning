"""
ESMC Embedding Library

A library to generate protein sequence embeddings using ESM-C models.
Designed to work with outputs from fasta_cleaner.py.

Usage:
    from esmc_embed_lib import load_esmc_model, embed_sequences, embed_from_csv
    
    # Load model
    model = load_esmc_model("your_hf_token")
    
    # Embed from DataFrame (defaults: last layer embeddings + logits)
    results = embed_sequences(model, sequences_df)
    
    # Extract specific hidden layers
    results = embed_sequences(model, sequences_df, hidden_layers=[12, 24, 36])
    
    # Extract all layers
    results = embed_sequences(model, sequences_df, hidden_layers="all")
    
    # Disable logits output
    results = embed_sequences(model, sequences_df, return_logits=False)
    
    # Save results
    save_embeddings(results, "embeddings.pt")
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
import torch

# ESM imports (will fail if not installed - handled gracefully)
try:
    from esm.models.esmc import ESMC, ESMCInferenceClient, LogitsConfig
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsOutput
    from huggingface_hub import login
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MODEL = "esmc_600m"
VALID_MODELS = ["esmc_300m", "esmc_600m"]

# Number of layers per model (for "all" option)
MODEL_LAYERS = {
    "esmc_300m": 36,
    "esmc_600m": 36,
}


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_esmc_model(
    hf_token: str,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto"
) -> ESMCInferenceClient:
    """
    Load an ESM-C model from HuggingFace.
    
    Args:
        hf_token: Your HuggingFace access token
        model_name: Model to load ("esmc_300m" or "esmc_600m")
        device: Device to load model on ("auto", "cuda", or "cpu")
        
    Returns:
        Loaded ESMCInferenceClient model
        
    Raises:
        ImportError: If ESM libraries are not installed
        ValueError: If model_name is not valid
        
    Example:
        >>> model = load_esmc_model("hf_abc123")
        >>> model = load_esmc_model("hf_abc123", model_name="esmc_300m", device="cpu")
    """
    if not ESM_AVAILABLE:
        raise ImportError(
            "ESM libraries not installed. Install with:\n"
            "pip install esm huggingface_hub"
        )
    
    if model_name not in VALID_MODELS:
        raise ValueError(f"model_name must be one of {VALID_MODELS}, got '{model_name}'")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Login to HuggingFace
    login(token=hf_token)
    
    # Load model
    model = ESMC.from_pretrained(model_name).to(device)
    
    return model


# =============================================================================
# SEQUENCE CLEANING
# =============================================================================

def clean_sequence(sequence: str) -> str:
    """
    Clean a sequence by removing non-alphabetic characters.
    
    Args:
        sequence: Raw amino acid sequence
        
    Returns:
        Cleaned sequence (uppercase, only letters)
    """
    return re.sub(r"[^A-Z]", "", sequence.upper())


def _convert_to_protein(sequence: str) -> ESMProtein:
    """
    Convert a sequence string to ESMProtein object.
    
    Args:
        sequence: Cleaned amino acid sequence
        
    Returns:
        ESMProtein object ready for embedding
    """
    cleaned = clean_sequence(sequence)
    return ESMProtein(
        sequence=cleaned,
        potential_sequence_of_concern=True
    )


# =============================================================================
# SINGLE EMBEDDING
# =============================================================================

def embed_single(
    model: ESMCInferenceClient,
    sequence: str,
    return_embeddings: bool = True,
    return_logits: bool = True,
    hidden_layers: Optional[Union[int, List[int], str]] = None
) -> Dict:
    """
    Embed a single protein sequence.
    
    Args:
        model: Loaded ESMC model
        sequence: Amino acid sequence to embed
        return_embeddings: Whether to return last-layer embeddings (default: True)
        return_logits: Whether to return logits (default: True)
        hidden_layers: Which hidden layer(s) to extract:
            - None: No hidden states (default)
            - int: Single layer index (e.g., -1 for last, 12 for layer 12)
            - List[int]: Multiple layers (e.g., [12, 24, 36])
            - "all": All layers
        
    Returns:
        Dictionary with 'embeddings', 'logits', 'hidden_states' keys
        
    Example:
        >>> output = embed_single(model, "MKTAYIAKQRQISFVK")
        >>> embeddings = output["embeddings"]
        
        >>> output = embed_single(model, "MKTAYIAKQRQISFVK", hidden_layers=[12, 24])
        >>> layer_12 = output["hidden_states"][12]
    """
    protein = _convert_to_protein(sequence)
    protein_tensor = model.encode(protein)
    
    result = {
        "embeddings": None,
        "logits": None,
        "hidden_states": {}
    }
    
    # Determine which layers to extract
    layers_to_extract = _normalize_hidden_layers(hidden_layers)
    
    # Get embeddings and logits with a single forward pass
    logits_config = LogitsConfig(
        sequence=return_logits,
        return_embeddings=return_embeddings,
        return_hidden_states=len(layers_to_extract) > 0,
        ith_hidden_layer=layers_to_extract[0] if len(layers_to_extract) == 1 else -1
    )
    
    output = model.logits(protein_tensor, logits_config)
    
    # Extract embeddings
    if return_embeddings and output.embeddings is not None:
        result["embeddings"] = output.embeddings.squeeze(0).detach().cpu()
    
    # Extract logits
    if return_logits and output.logits is not None:
        result["logits"] = output.logits.sequence.squeeze(0).detach().cpu()
    
    # Extract hidden states (may need multiple passes for multiple layers)
    if len(layers_to_extract) == 1:
        hs = getattr(output, "hidden_states", None)
        if isinstance(hs, torch.Tensor):
            result["hidden_states"][layers_to_extract[0]] = hs.squeeze().detach().cpu()
    elif len(layers_to_extract) > 1:
        # Multiple layers require multiple forward passes
        for layer_idx in layers_to_extract:
            layer_config = LogitsConfig(
                sequence=False,
                return_embeddings=False,
                return_hidden_states=True,
                ith_hidden_layer=layer_idx
            )
            layer_output = model.logits(protein_tensor, layer_config)
            hs = getattr(layer_output, "hidden_states", None)
            if isinstance(hs, torch.Tensor):
                result["hidden_states"][layer_idx] = hs.squeeze().detach().cpu()
    
    return result


def _normalize_hidden_layers(
    hidden_layers: Optional[Union[int, List[int], str]],
    model_name: str = "esmc_600m"
) -> List[int]:
    """
    Normalize hidden_layers parameter to a list of layer indices.
    
    Args:
        hidden_layers: User-specified layer(s)
        model_name: Model name (to determine total layers for "all")
        
    Returns:
        List of layer indices to extract
    """
    if hidden_layers is None:
        return []
    
    if isinstance(hidden_layers, int):
        return [hidden_layers]
    
    if isinstance(hidden_layers, str):
        if hidden_layers.lower() == "all":
            total = MODEL_LAYERS.get(model_name, 36)
            return list(range(1, total + 1))
        else:
            raise ValueError(f"Invalid hidden_layers string: '{hidden_layers}'. Use 'all' or a list of ints.")
    
    if isinstance(hidden_layers, (list, tuple)):
        return list(hidden_layers)
    
    raise ValueError(f"hidden_layers must be int, list of ints, 'all', or None. Got: {type(hidden_layers)}")


# =============================================================================
# BATCH EMBEDDING
# =============================================================================

def embed_sequences(
    model: ESMCInferenceClient,
    sequences_df: pd.DataFrame,
    return_embeddings: bool = True,
    return_logits: bool = True,
    hidden_layers: Optional[Union[int, List[int], str]] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Embed multiple protein sequences from a DataFrame.
    
    Expects a DataFrame with columns:
    - sequence_id: Unique identifier for each sequence
    - sequence: Amino acid sequence string
    
    These columns match the output format of fasta_cleaner.py.
    
    **Default behavior (no parameters specified):**
    Returns last-layer embeddings and logits for each sequence.
    
    Args:
        model: Loaded ESMC model
        sequences_df: DataFrame with 'sequence_id' and 'sequence' columns
        return_embeddings: Whether to return last-layer embeddings (default: True)
        return_logits: Whether to return logits (default: True)
        hidden_layers: Which hidden layer(s) to extract:
            - None: No hidden states (default)
            - int: Single layer index (e.g., -1 for last, 12 for layer 12)
            - List[int]: Multiple layers (e.g., [12, 24, 36])
            - "all": All layers (warning: memory intensive!)
        max_workers: Max parallel threads (None = default)
        progress_callback: Optional function(current, total) for progress updates
        
    Returns:
        Dictionary with:
        - sequence_id: List of sequence IDs
        - embeddings: List of embedding tensors (or None if disabled)
        - logits: List of logits tensors (or None if disabled)
        - hidden_states: List of dicts {layer_idx: tensor} for each sequence
        - hidden_layers_extracted: List of layer indices that were extracted
        - model_name: Name of model used
        - created_at: Timestamp
        - errors: List of (sequence_id, error_message) tuples
        
    Raises:
        ValueError: If required columns are missing
        
    Example:
        # Default: embeddings + logits
        >>> results = embed_sequences(model, sequences_df)
        
        # Extract specific hidden layers
        >>> results = embed_sequences(model, sequences_df, hidden_layers=[12, 24, 36])
        >>> layer_12 = results["hidden_states"][0][12]  # First sequence, layer 12
        
        # Embeddings only, no logits
        >>> results = embed_sequences(model, sequences_df, return_logits=False)
    """
    # Validate input
    required_cols = {"sequence_id", "sequence"}
    if not required_cols.issubset(sequences_df.columns):
        missing = required_cols - set(sequences_df.columns)
        raise ValueError(
            f"DataFrame missing required columns: {missing}\n"
            f"Expected columns from fasta_cleaner.py output."
        )
    
    # Normalize hidden layers
    layers_to_extract = _normalize_hidden_layers(hidden_layers)
    
    # Extract data
    sequence_ids = sequences_df["sequence_id"].tolist()
    sequences = sequences_df["sequence"].tolist()
    total = len(sequences)
    
    # Initialize results
    results = {
        "sequence_id": [],
        "embeddings": [],
        "logits": [],
        "hidden_states": [],
        "hidden_layers_extracted": layers_to_extract,
        "model_name": str(model.__class__.__name__),
        "created_at": datetime.now().isoformat(),
        "errors": [],
        "config": {
            "return_embeddings": return_embeddings,
            "return_logits": return_logits,
            "hidden_layers": hidden_layers
        }
    }
    
    def process_one(args):
        idx, seq_id, seq = args
        try:
            protein = _convert_to_protein(seq)
            protein_tensor = model.encode(protein)
            
            # Main forward pass for embeddings and logits
            main_config = LogitsConfig(
                sequence=return_logits,
                return_embeddings=return_embeddings,
                return_hidden_states=len(layers_to_extract) == 1,
                ith_hidden_layer=layers_to_extract[0] if len(layers_to_extract) == 1 else -1
            )
            output = model.logits(protein_tensor, main_config)
            
            # Build result dict for this sequence
            seq_result = {
                "embeddings": None,
                "logits": None,
                "hidden_states": {}
            }
            
            # Extract embeddings
            if return_embeddings and output.embeddings is not None:
                seq_result["embeddings"] = output.embeddings.squeeze(0).detach().cpu()
            
            # Extract logits
            if return_logits and output.logits is not None:
                seq_result["logits"] = output.logits.sequence.squeeze(0).detach().cpu()
            
            # Extract hidden states
            if len(layers_to_extract) == 1:
                hs = getattr(output, "hidden_states", None)
                if isinstance(hs, torch.Tensor):
                    seq_result["hidden_states"][layers_to_extract[0]] = hs.squeeze().detach().cpu()
            elif len(layers_to_extract) > 1:
                # Multiple layers require multiple forward passes
                for layer_idx in layers_to_extract:
                    layer_config = LogitsConfig(
                        sequence=False,
                        return_embeddings=False,
                        return_hidden_states=True,
                        ith_hidden_layer=layer_idx
                    )
                    layer_output = model.logits(protein_tensor, layer_config)
                    hs = getattr(layer_output, "hidden_states", None)
                    if isinstance(hs, torch.Tensor):
                        seq_result["hidden_states"][layer_idx] = hs.squeeze().detach().cpu()
            
            return (idx, seq_id, seq_result, None)
        except Exception as e:
            return (idx, seq_id, None, str(e))
    
    # Process sequences
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_one, (i, sid, seq))
            for i, (sid, seq) in enumerate(zip(sequence_ids, sequences))
        ]
        
        completed = 0
        for future in futures:
            idx, seq_id, seq_result, error = future.result()
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total)
            
            if error:
                results["errors"].append((seq_id, error))
                results["sequence_id"].append(seq_id)
                results["logits"].append(None)
                results["embeddings"].append(None)
                results["hidden_states"].append({})
            else:
                results["sequence_id"].append(seq_id)
                results["embeddings"].append(seq_result["embeddings"])
                results["logits"].append(seq_result["logits"])
                results["hidden_states"].append(seq_result["hidden_states"])
    
    return results


def embed_from_csv(
    model: ESMCInferenceClient,
    sequences_path: Union[str, Path],
    return_embeddings: bool = True,
    return_logits: bool = True,
    hidden_layers: Optional[Union[int, List[int], str]] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Embed sequences from a CSV file (output of fasta_cleaner.py).
    
    **Default behavior (no parameters specified):**
    Returns last-layer embeddings and logits for each sequence.
    
    Args:
        model: Loaded ESMC model
        sequences_path: Path to sequences.csv from fasta_cleaner.py
        return_embeddings: Whether to return last-layer embeddings (default: True)
        return_logits: Whether to return logits (default: True)
        hidden_layers: Which hidden layer(s) to extract:
            - None: No hidden states (default)
            - int: Single layer index (e.g., -1 for last, 12 for layer 12)
            - List[int]: Multiple layers (e.g., [12, 24, 36])
            - "all": All layers
        max_workers: Max parallel threads (None = default)
        progress_callback: Optional function(current, total) for progress updates
        
    Returns:
        Dictionary with embedding results (see embed_sequences)
        
    Example:
        # Default: embeddings + logits
        >>> results = embed_from_csv(model, "sequences.csv")
        
        # Extract layers 12, 24, 36
        >>> results = embed_from_csv(model, "sequences.csv", hidden_layers=[12, 24, 36])
        
        # No logits
        >>> results = embed_from_csv(model, "sequences.csv", return_logits=False)
    """
    sequences_df = pd.read_csv(sequences_path, keep_default_na=False)
    
    return embed_sequences(
        model=model,
        sequences_df=sequences_df,
        return_embeddings=return_embeddings,
        return_logits=return_logits,
        hidden_layers=hidden_layers,
        max_workers=max_workers,
        progress_callback=progress_callback
    )


# =============================================================================
# SAVING RESULTS
# =============================================================================

def save_embeddings(
    results: Dict,
    output_path: Union[str, Path]
) -> Path:
    """
    Save embedding results to a PyTorch .pt file.
    
    Args:
        results: Dictionary from embed_sequences or embed_from_csv
        output_path: Path to save the .pt file
        
    Returns:
        Path to the saved file
        
    Example:
        >>> save_embeddings(results, "embeddings.pt")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(results, output_path)
    
    return output_path


def load_embeddings(path: Union[str, Path]) -> Dict:
    """
    Load embedding results from a .pt file.
    
    Args:
        path: Path to the .pt file
        
    Returns:
        Dictionary with embedding results
        
    Example:
        >>> results = load_embeddings("embeddings.pt")
        >>> embeddings = results["embeddings"]
    """
    return torch.load(path, weights_only=False)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_embedding_for_sequence(
    results: Dict,
    sequence_id: str
) -> Optional[torch.Tensor]:
    """
    Get the embedding for a specific sequence ID.
    
    Args:
        results: Dictionary from embed_sequences
        sequence_id: The sequence ID to look up
        
    Returns:
        Embedding tensor or None if not found
        
    Example:
        >>> emb = get_embedding_for_sequence(results, "99603f8fb1e9")
    """
    try:
        idx = results["sequence_id"].index(sequence_id)
        return results["embeddings"][idx]
    except ValueError:
        return None


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """
    Convert results to a summary DataFrame.
    
    Args:
        results: Dictionary from embed_sequences
        
    Returns:
        DataFrame with sequence_id, embedding_shape, has_error columns
        
    Example:
        >>> df = results_to_dataframe(results)
        >>> print(df)
    """
    data = []
    for i, seq_id in enumerate(results["sequence_id"]):
        emb = results["embeddings"][i]
        has_error = any(seq_id == err[0] for err in results.get("errors", []))
        
        data.append({
            "sequence_id": seq_id,
            "embedding_shape": tuple(emb.shape) if emb is not None else None,
            "has_error": has_error
        })
    
    return pd.DataFrame(data)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("ESMC Embedding Library")
    print("=" * 50)
    print("\nThis is a library module. Import it in your code:")
    print("\n  from esmc_embed_lib import load_esmc_model, embed_from_csv")
    print("\n  model = load_esmc_model('your_hf_token')")
    print("  results = embed_from_csv(model, 'sequences.csv')")
    print("\nFor interactive use, see esmc_embed_colab.ipynb")
