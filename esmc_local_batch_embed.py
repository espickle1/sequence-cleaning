"""
ESMC Local Batch Embed (Simplified)

A streamlined script for batch embedding protein sequences using ESM-C models.
Accepts ONLY CSV output from fasta_cleaner.py as input.

For more features, use:
- esmc_embed_lib.py (Python library with function API)
- esmc_embed_colab.ipynb (Interactive Jupyter notebook)

Usage:
    This script uses a YAML config file. Example config:
    
    paths:
      input_path: "sequences.csv"    # From fasta_cleaner.py
      output_path: "embeddings.pt"
    
    parameters:
      token: "hf_your_token"
      model_name: "esmc_600m"
      return_embeddings: true        # Last layer embeddings (default: true)
      return_logits: true            # Logits output (default: true)
      hidden_layers: null            # Options: null, -1, [12, 24, 36], "all"

Run:
    python esmc_local_batch_embed.py config.yaml
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
import yaml

# ESM libraries
from esm.models.esmc import ESMC, ESMCInferenceClient, LogitsConfig
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsOutput
from huggingface_hub import login


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_LAYERS = {"esmc_300m": 36, "esmc_600m": 36}


# =============================================================================
# CONFIGURATION
# =============================================================================

class EmbedConfig:
    """Simple configuration loader for embedding."""
    
    def __init__(self, config_path: str | Path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Paths
        paths = cfg.get("paths", {})
        self.input_path = Path(paths.get("input_path", "sequences.csv"))
        self.output_path = Path(paths.get("output_path", "embeddings.pt"))
        
        # Parameters
        params = cfg.get("parameters", {})
        self.token = params.get("token", "")
        self.model_name = params.get("model_name", "esmc_600m")
        self.return_embeddings = bool(params.get("return_embeddings", True))
        self.return_logits = bool(params.get("return_logits", True))
        self.max_workers = params.get("max_workers", None)
        
        # Hidden layers - can be: null, int, list of ints, or "all"
        hidden = params.get("hidden_layers", None)
        self.hidden_layers = self._normalize_hidden_layers(hidden)
    
    def _normalize_hidden_layers(self, hidden) -> List[int]:
        """Normalize hidden_layers config to list of ints."""
        if hidden is None:
            return []
        if isinstance(hidden, int):
            return [hidden]
        if isinstance(hidden, str) and hidden.lower() == "all":
            return list(range(1, MODEL_LAYERS.get(self.model_name, 36) + 1))
        if isinstance(hidden, list):
            return [int(x) for x in hidden]
        return []



# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(token: str, model_name: str = "esmc_600m") -> ESMCInferenceClient:
    """Load ESM-C model from HuggingFace."""
    login(token=token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained(model_name).to(device)
    print(f"Model {model_name} loaded on {device}")
    return model


# =============================================================================
# SEQUENCE PROCESSING
# =============================================================================

def clean_sequence(seq: str) -> str:
    """Remove non-alphabetic characters from sequence."""
    return re.sub(r"[^A-Z]", "", seq.upper())


def convert_sequence(sequence: str) -> ESMProtein:
    """Convert sequence string to ESMProtein object."""
    cleaned = clean_sequence(sequence)
    return ESMProtein(
        sequence=cleaned,
        potential_sequence_of_concern=True
    )


# =============================================================================
# CSV LOADING (fasta_cleaner.py format)
# =============================================================================

def load_sequences_csv(input_path: str | Path) -> tuple[list[str], pd.DataFrame]:
    """
    Load sequences from CSV file (fasta_cleaner.py output format).
    
    Expected columns:
    - sequence_id: Unique identifier
    - sequence: Amino acid sequence
    - length: Sequence length (optional)
    
    Returns:
        Tuple of (sequence_list, dataframe)
    """
    df = pd.read_csv(input_path, keep_default_na=False)
    
    # Validate required columns
    required = {"sequence_id", "sequence"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")
    
    sequences = df["sequence"].tolist()
    return sequences, df


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def embed_single_sequence(
    model: ESMCInferenceClient, 
    sequence: str, 
    logits_config: LogitsConfig
) -> LogitsOutput:
    """Embed a single sequence and return LogitsOutput."""
    protein = convert_sequence(sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, logits_config)
    return output


def batch_embed_sequences(
    model: ESMCInferenceClient, 
    sequences: Sequence[str],
    logits_config: LogitsConfig,
    max_workers: Optional[int] = None
) -> Sequence[LogitsOutput | ESMProteinError]:
    """
    Batch embed multiple sequences using ThreadPoolExecutor.
    
    Args:
        model: The ESMC inference client
        sequences: List of protein sequences to embed
        logits_config: Configuration for logits output
        max_workers: Maximum number of threads (None = default)
    
    Returns:
        List of LogitsOutput objects or ESMProteinError for failed embeddings
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(embed_single_sequence, model, seq, logits_config) 
            for seq in sequences
        ]
        results = []
        for i, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
            
            # Progress update
            if (i + 1) % 10 == 0 or i + 1 == len(futures):
                print(f"  Processed {i + 1}/{len(futures)} sequences")
    
    return results


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

def run_inference(config_path: str | Path) -> Dict:
    """
    Run batch embedding using configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with embedding results
    """
    # Load configuration
    cfg = EmbedConfig(config_path)
    print(f"Configuration loaded from {config_path}")
    print(f"  • Embeddings: {cfg.return_embeddings}")
    print(f"  • Logits: {cfg.return_logits}")
    if cfg.hidden_layers:
        print(f"  • Hidden layers: {cfg.hidden_layers}")
    
    # Load model
    model = load_model(cfg.token, cfg.model_name)
    
    # Load sequences from CSV
    sequences_list, sequences_df = load_sequences_csv(cfg.input_path)
    ids = sequences_df["sequence_id"].tolist()
    print(f"Processing {len(sequences_list)} sequences from {cfg.input_path}")
    
    # Initialize results
    results = {
        "sequence_id": [],
        "logits": [],
        "embeddings": [],
        "hidden_states": [],
        "hidden_layers_extracted": cfg.hidden_layers,
        "model_name": cfg.model_name,
        "created_at": datetime.now().isoformat(),
        "errors": [],
        "config": {
            "return_embeddings": cfg.return_embeddings,
            "return_logits": cfg.return_logits,
            "hidden_layers": cfg.hidden_layers if cfg.hidden_layers else None
        }
    }
    
    # Process each sequence
    for i, (seq_id, seq) in enumerate(zip(ids, sequences_list)):
        if (i + 1) % 10 == 0 or i + 1 == len(ids):
            print(f"  Processed {i + 1}/{len(ids)} sequences")
        
        try:
            protein = convert_sequence(seq)
            protein_tensor = model.encode(protein)
            
            # Main forward pass for embeddings and logits
            main_config = LogitsConfig(
                sequence=cfg.return_logits,
                return_embeddings=cfg.return_embeddings,
                return_hidden_states=len(cfg.hidden_layers) == 1,
                ith_hidden_layer=cfg.hidden_layers[0] if len(cfg.hidden_layers) == 1 else -1
            )
            output = model.logits(protein_tensor, main_config)
            
            # Build result for this sequence
            seq_hidden = {}
            
            results["sequence_id"].append(seq_id)
            
            # Logits
            if cfg.return_logits and output.logits is not None:
                results["logits"].append(
                    output.logits.sequence.squeeze(0).detach().cpu()
                )
            else:
                results["logits"].append(None)
            
            # Embeddings
            if cfg.return_embeddings and output.embeddings is not None:
                results["embeddings"].append(
                    output.embeddings.squeeze(0).detach().cpu()
                )
            else:
                results["embeddings"].append(None)
            
            # Hidden states
            if len(cfg.hidden_layers) == 1:
                hs = getattr(output, "hidden_states", None)
                if isinstance(hs, torch.Tensor):
                    seq_hidden[cfg.hidden_layers[0]] = hs.squeeze().detach().cpu()
            elif len(cfg.hidden_layers) > 1:
                # Multiple layers require multiple forward passes
                for layer_idx in cfg.hidden_layers:
                    layer_config = LogitsConfig(
                        sequence=False,
                        return_embeddings=False,
                        return_hidden_states=True,
                        ith_hidden_layer=layer_idx
                    )
                    layer_output = model.logits(protein_tensor, layer_config)
                    hs = getattr(layer_output, "hidden_states", None)
                    if isinstance(hs, torch.Tensor):
                        seq_hidden[layer_idx] = hs.squeeze().detach().cpu()
            
            results["hidden_states"].append(seq_hidden)
            
        except Exception as e:
            print(f"Error processing {seq_id}: {e}")
            results["errors"].append((seq_id, str(e)))
            results["sequence_id"].append(seq_id)
            results["logits"].append(None)
            results["embeddings"].append(None)
            results["hidden_states"].append({})
    
    # Save results
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output_path)
    
    # Summary
    error_count = len(results["errors"])
    success_count = len(results["sequence_id"]) - error_count
    print(f"\nResults saved to {cfg.output_path}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    
    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python esmc_local_batch_embed.py <config.yaml>")
        print("\nExample config.yaml:")
        print("""
paths:
  input_path: "sequences.csv"    # From fasta_cleaner.py
  output_path: "embeddings.pt"

parameters:
  token: "hf_your_token"
  model_name: "esmc_600m"
  return_embeddings: true        # Last layer embeddings (default: true)
  return_logits: true            # Logits output (default: true)
  hidden_layers: null            # Options: null, -1, [12, 24, 36], "all"
""")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_inference(config_path)
