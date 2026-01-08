"""
Entropy Analysis Library

Calculate Shannon entropy from protein sequence logits to identify
conserved (low entropy) and variable (high entropy) positions.

Usage:
    from analysis.entropy_lib import analyze_entropy, entropy_summary
    
    # From embeddings.pt file
    results = torch.load("embeddings.pt")
    entropy_results = analyze_entropy(results)
    
    # Get summary DataFrame
    df = entropy_summary(entropy_results)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


# =============================================================================
# CORE ENTROPY CALCULATION
# =============================================================================

def calculate_entropy(
    logits: torch.Tensor,
    base: str = "e"
) -> torch.Tensor:
    """
    Calculate Shannon entropy for each residue from logits.
    
    Args:
        logits: Tensor of shape (num_residues, num_tokens)
        base: Entropy base - 'e' (nats), '2' (bits), or '10' (dits)
        
    Returns:
        Tensor of shape (num_residues,) with entropy values
        
    Formula: H(X) = -Σ p(x) * log(p(x))
    
    Example:
        >>> entropy = calculate_entropy(logits)
        >>> mean_entropy = entropy.mean()
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Calculate entropy
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # Convert base if needed
    if base == "2":
        entropy = entropy / np.log(2)
    elif base == "10":
        entropy = entropy / np.log(10)
    
    return entropy


def calculate_entropy_batched(
    logits: torch.Tensor,
    base: str = "e",
    batch_size: int = 10000,
    use_mixed_precision: bool = False
) -> torch.Tensor:
    """
    Calculate Shannon entropy for large tensors using batching.
    
    More memory-efficient for very long sequences or many sequences.
    
    Args:
        logits: Tensor of shape (num_residues, num_tokens)
        base: Entropy base - 'e' (nats), '2' (bits), or '10' (dits)
        batch_size: Residues per batch (adjust based on GPU memory)
        use_mixed_precision: If True, use fp16 for faster computation
        
    Returns:
        Tensor of shape (num_residues,) with entropy values
        
    Example:
        >>> entropy = calculate_entropy_batched(logits, batch_size=5000)
    """
    num_residues = logits.shape[0]
    entropy_list = []
    
    # Optionally convert to fp16
    if use_mixed_precision:
        logits = logits.half()
    
    for i in range(0, num_residues, batch_size):
        batch_end = min(i + batch_size, num_residues)
        logits_batch = logits[i:batch_end]
        
        # Process batch
        probs_batch = torch.softmax(logits_batch, dim=-1)
        epsilon = 1e-10
        log_probs_batch = torch.log(probs_batch + epsilon)
        entropy_batch = -torch.sum(probs_batch * log_probs_batch, dim=-1)
        
        # Convert back to fp32 if mixed precision
        if use_mixed_precision:
            entropy_batch = entropy_batch.float()
        
        entropy_list.append(entropy_batch)
    
    # Concatenate batches
    entropy = torch.cat(entropy_list, dim=0)
    
    # Convert base
    if base == "2":
        entropy = entropy / np.log(2)
    elif base == "10":
        entropy = entropy / np.log(10)
    
    return entropy


# =============================================================================
# POSITION CLASSIFICATION
# =============================================================================

def get_constrained_positions(
    entropy: torch.Tensor,
    percentile: float = 10.0
) -> torch.Tensor:
    """
    Find conserved/constrained positions (low entropy).
    
    Args:
        entropy: Tensor of entropy values
        percentile: Percentile threshold (positions below this are constrained)
        
    Returns:
        Tensor of indices for constrained positions
        
    Example:
        >>> constrained = get_constrained_positions(entropy, percentile=10)
        >>> print(f"Found {len(constrained)} conserved positions")
    """
    entropy = entropy.float()
    threshold = torch.quantile(entropy, percentile / 100.0)
    return torch.where(entropy < threshold)[0]


def get_flexible_positions(
    entropy: torch.Tensor,
    percentile: float = 90.0
) -> torch.Tensor:
    """
    Find variable/flexible positions (high entropy).
    
    Args:
        entropy: Tensor of entropy values
        percentile: Percentile threshold (positions above this are flexible)
        
    Returns:
        Tensor of indices for flexible positions
        
    Example:
        >>> flexible = get_flexible_positions(entropy, percentile=90)
        >>> print(f"Found {len(flexible)} variable positions")
    """
    entropy = entropy.float()
    threshold = torch.quantile(entropy, percentile / 100.0)
    return torch.where(entropy > threshold)[0]


# =============================================================================
# HIGH-LEVEL ANALYSIS
# =============================================================================

def analyze_entropy(
    results: Dict,
    base: str = "e",
    batch_size: int = 10000,
    constrained_percentile: float = 10.0,
    flexible_percentile: float = 90.0
) -> Dict:
    """
    Analyze entropy from embedding results (embeddings.pt format).
    
    Processes the logits from each sequence and calculates entropy,
    identifying conserved and variable positions.
    
    Args:
        results: Dictionary from embed_sequences (with 'logits' key)
        base: Entropy base - 'e', '2', or '10'
        batch_size: Batch size for large tensors
        constrained_percentile: Percentile for conserved positions
        flexible_percentile: Percentile for variable positions
        
    Returns:
        Dictionary with:
        - sequence_id: List of sequence IDs
        - entropy: List of per-residue entropy tensors
        - mean_entropy: List of mean entropy values
        - constrained_positions: List of conserved position indices
        - flexible_positions: List of variable position indices
        - global_mean: Overall mean entropy across all sequences
        
    Example:
        >>> results = torch.load("embeddings.pt")
        >>> entropy_results = analyze_entropy(results)
        >>> print(f"Global mean entropy: {entropy_results['global_mean']:.3f}")
    """
    sequence_ids = results.get("sequence_id", [])
    logits_list = results.get("logits", [])
    
    entropy_results = {
        "sequence_id": [],
        "entropy": [],
        "mean_entropy": [],
        "std_entropy": [],
        "min_entropy": [],
        "max_entropy": [],
        "constrained_positions": [],
        "flexible_positions": [],
        "num_residues": [],
    }
    
    all_entropies = []
    
    for seq_id, logits in zip(sequence_ids, logits_list):
        if logits is None:
            continue
        
        # Calculate entropy
        if logits.shape[0] > batch_size:
            entropy = calculate_entropy_batched(logits, base=base, batch_size=batch_size)
        else:
            entropy = calculate_entropy(logits, base=base)
        
        # Get positions
        constrained = get_constrained_positions(entropy, constrained_percentile)
        flexible = get_flexible_positions(entropy, flexible_percentile)
        
        # Store results
        entropy_results["sequence_id"].append(seq_id)
        entropy_results["entropy"].append(entropy.cpu())
        entropy_results["mean_entropy"].append(entropy.mean().item())
        entropy_results["std_entropy"].append(entropy.std().item())
        entropy_results["min_entropy"].append(entropy.min().item())
        entropy_results["max_entropy"].append(entropy.max().item())
        entropy_results["constrained_positions"].append(constrained.cpu())
        entropy_results["flexible_positions"].append(flexible.cpu())
        entropy_results["num_residues"].append(len(entropy))
        
        all_entropies.append(entropy)
    
    # Global statistics
    if all_entropies:
        combined = torch.cat(all_entropies)
        entropy_results["global_mean"] = combined.mean().item()
        entropy_results["global_std"] = combined.std().item()
    else:
        entropy_results["global_mean"] = None
        entropy_results["global_std"] = None
    
    return entropy_results


def entropy_summary(entropy_results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from entropy analysis results.
    
    Args:
        entropy_results: Output from analyze_entropy()
        
    Returns:
        DataFrame with per-sequence entropy statistics
        
    Example:
        >>> df = entropy_summary(entropy_results)
        >>> print(df.sort_values("mean_entropy").head())
    """
    data = {
        "sequence_id": entropy_results["sequence_id"],
        "num_residues": entropy_results["num_residues"],
        "mean_entropy": entropy_results["mean_entropy"],
        "std_entropy": entropy_results["std_entropy"],
        "min_entropy": entropy_results["min_entropy"],
        "max_entropy": entropy_results["max_entropy"],
        "num_constrained": [len(c) for c in entropy_results["constrained_positions"]],
        "num_flexible": [len(f) for f in entropy_results["flexible_positions"]],
    }
    
    return pd.DataFrame(data)


def save_entropy_results(
    entropy_results: Dict,
    output_path: str,
    include_tensors: bool = True
) -> None:
    """
    Save entropy analysis results.
    
    Args:
        entropy_results: Output from analyze_entropy()
        output_path: Path to save (.pt for tensors, .csv for summary)
        include_tensors: If True, save full tensors; if False, save summary only
        
    Example:
        >>> save_entropy_results(entropy_results, "entropy_results.pt")
        >>> save_entropy_results(entropy_results, "entropy_summary.csv", include_tensors=False)
    """
    if output_path.endswith(".csv"):
        df = entropy_summary(entropy_results)
        df.to_csv(output_path, index=False)
    else:
        if not include_tensors:
            # Remove tensor data for smaller file
            results_copy = {k: v for k, v in entropy_results.items() 
                          if k not in ["entropy", "constrained_positions", "flexible_positions"]}
            torch.save(results_copy, output_path)
        else:
            torch.save(entropy_results, output_path)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_batch_size(
    num_residues: int,
    num_tokens: int = 64,
    gpu_memory_gb: float = 8.0
) -> int:
    """
    Estimate appropriate batch size based on available GPU memory.
    
    Args:
        num_residues: Number of residues to process
        num_tokens: Number of token types (vocabulary size)
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        Recommended batch size
    """
    bytes_per_element = 4  # fp32
    intermediate_factor = 4  # logits, probs, log_probs, entropy
    
    available_bytes = gpu_memory_gb * 1e9 * 0.8  # Use 80% of memory
    bytes_per_residue = num_tokens * bytes_per_element * intermediate_factor
    
    batch_size = int(available_bytes / bytes_per_residue)
    batch_size = max(1000, min(batch_size, 50000))
    
    return batch_size
