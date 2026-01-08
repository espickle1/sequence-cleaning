"""
Logits Analysis Library

Analyze and visualize protein sequence logits with pooling, scaling, 
and amino acid propensity heatmaps.

Usage:
    from analysis.logits_lib import analyze_residues, plot_heatmap
    
    # Analyze specific residues
    results = torch.load("embeddings.pt")
    analysis = analyze_residues(results, residues_of_interest={100: "R100", 200: "K200"})
    
    # Generate heatmap
    plot_heatmap(analysis["scaled_logits"], analysis["residue_labels"], AA_VOCAB)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard amino acid vocabulary (single letter codes)
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")

# Extended vocabulary matching ESM models (includes special tokens)
ESM_VOCAB_FULL = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
    "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
    "X", "B", "U", "Z", "O", ".", "-",
    "<mask>"
]

# Amino acid indices in ESM vocabulary (positions 4-23)
ESM_AA_START = 4
ESM_AA_END = 24


# =============================================================================
# LOGITS POOLING
# =============================================================================

def pool_logits(
    logits_list: List[torch.Tensor],
    indices: Optional[List[int]] = None,
    method: str = "mean"
) -> torch.Tensor:
    """
    Pool logits across multiple sequences or positions.
    
    Args:
        logits_list: List of logit tensors to pool
        indices: Optional indices to select before pooling
        method: Pooling method - 'mean', 'max', or 'sum'
        
    Returns:
        Pooled logits tensor
        
    Example:
        >>> pooled = pool_logits(results["logits"], method="mean")
    """
    if indices is not None:
        logits_list = [logits_list[i] for i in indices if i < len(logits_list)]
    
    if not logits_list:
        raise ValueError("No logits to pool")
    
    # Stack tensors
    stacked = torch.stack(logits_list, dim=0)
    
    # Apply pooling
    if method == "mean":
        return stacked.mean(dim=0)
    elif method == "max":
        return stacked.max(dim=0)[0]
    elif method == "sum":
        return stacked.sum(dim=0)
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def extract_amino_acid_probs(
    logits: torch.Tensor,
    vocab: List[str] = None,
    aa_start: int = ESM_AA_START,
    aa_end: int = ESM_AA_END
) -> pd.DataFrame:
    """
    Extract amino acid probabilities from logits.
    
    Converts logits to probabilities and returns a DataFrame
    with amino acid columns.
    
    Args:
        logits: Logits tensor of shape (num_residues, vocab_size)
        vocab: Amino acid vocabulary (if None, uses standard 20 AAs)
        aa_start: Start index of amino acids in vocabulary
        aa_end: End index of amino acids in vocabulary
        
    Returns:
        DataFrame with shape (num_residues, num_amino_acids)
        
    Example:
        >>> probs_df = extract_amino_acid_probs(logits)
        >>> print(probs_df.head())
    """
    if vocab is None:
        vocab = AA_VOCAB
    
    # Extract amino acid logits
    aa_logits = logits[:, aa_start:aa_end]
    
    # Convert to probabilities
    probs = torch.softmax(aa_logits, dim=-1)
    
    # Convert to DataFrame
    probs_np = probs.float().cpu().numpy()
    
    return pd.DataFrame(probs_np, columns=vocab[:probs_np.shape[1]])


# =============================================================================
# SCALING
# =============================================================================

def scale_logits(
    logits: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    method: str = "minmax"
) -> np.ndarray:
    """
    Scale logits using various normalization methods.
    
    Args:
        logits: Logits data (DataFrame, ndarray, or Tensor)
        method: Scaling method:
            - 'minmax': Scale to [0, 1] range
            - 'robust': Scale using median/IQR (robust to outliers)
            - 'power': Yeo-Johnson power transform
            - 'standard': Standardize to mean=0, std=1
            - 'softmax': Apply softmax normalization
            
    Returns:
        Scaled logits as numpy array
        
    Example:
        >>> scaled = scale_logits(logits_df, method="robust")
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        data = logits.float().cpu().numpy()
    elif isinstance(logits, pd.DataFrame):
        data = logits.values
    else:
        data = np.array(logits)
    
    original_shape = data.shape
    
    if method == "softmax":
        # Apply softmax row-wise
        exp_data = np.exp(data - np.max(data, axis=-1, keepdims=True))
        return exp_data / np.sum(exp_data, axis=-1, keepdims=True)
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for scaling methods. Install with: pip install scikit-learn")
    
    # Select scaler
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "power":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}. Choose from 'minmax', 'robust', 'power', 'standard', 'softmax'")
    
    # Scale column-wise (transpose, scale, transpose back)
    scaled = scaler.fit_transform(data.T).T
    
    return scaled.reshape(original_shape)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_heatmap(
    data: Union[np.ndarray, pd.DataFrame],
    row_labels: List[str] = None,
    col_labels: List[str] = None,
    title: str = "Amino Acid Propensity",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None,
    save_path: str = None
) -> None:
    """
    Generate a heatmap visualization of logits/probabilities.
    
    Args:
        data: 2D array of values to plot
        row_labels: Labels for rows (residues)
        col_labels: Labels for columns (amino acids)
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Matplotlib colormap
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        save_path: If provided, save figure to this path
        
    Example:
        >>> plot_heatmap(scaled_data, residue_labels, AA_VOCAB)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")
    
    # Convert to numpy if needed
    if isinstance(data, pd.DataFrame):
        if col_labels is None:
            col_labels = list(data.columns)
        if row_labels is None:
            row_labels = list(data.index)
        data = data.values
    
    # Default labels
    if col_labels is None:
        col_labels = AA_VOCAB[:data.shape[1]]
    if row_labels is None:
        row_labels = [str(i) for i in range(data.shape[0])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Propensity", fontsize=10)
    
    # Labels
    ax.set_xlabel("Amino Acid", fontsize=11)
    ax.set_ylabel("Residue", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


# =============================================================================
# HIGH-LEVEL ANALYSIS
# =============================================================================

def analyze_residues(
    results: Dict,
    residues_of_interest: Dict[int, str] = None,
    sequence_indices: List[int] = None,
    pool_method: str = "mean",
    scale_method: str = "minmax",
    vocab: List[str] = None
) -> Dict:
    """
    Analyze logits for specific residues of interest.
    
    Args:
        results: Dictionary from embed_sequences (with 'logits' key)
        residues_of_interest: Dict mapping residue positions to labels
                             e.g., {100: "R100", 200: "K200"}
        sequence_indices: Optional indices of sequences to analyze
                         (None = all sequences)
        pool_method: How to pool across sequences ('mean', 'max', 'sum')
        scale_method: Scaling method for visualization
        vocab: Amino acid vocabulary (None = standard 20 AAs)
        
    Returns:
        Dictionary with:
        - logits_pooled: Pooled logits DataFrame
        - probs: Probability DataFrame
        - scaled_logits: Scaled logits for visualization
        - residue_labels: Labels for residues
        
    Example:
        >>> analysis = analyze_residues(
        ...     results,
        ...     residues_of_interest={100: "D100", 200: "E200"}
        ... )
        >>> plot_heatmap(analysis["scaled_logits"], analysis["residue_labels"])
    """
    if vocab is None:
        vocab = AA_VOCAB
    
    logits_list = results.get("logits", [])
    
    # Filter sequences
    if sequence_indices is not None:
        logits_list = [logits_list[i] for i in sequence_indices if i < len(logits_list)]
    
    # Remove None entries
    logits_list = [l for l in logits_list if l is not None]
    
    if not logits_list:
        raise ValueError("No valid logits found")
    
    # Pool across sequences
    pooled = pool_logits(logits_list, method=pool_method)
    
    # Extract amino acid probabilities
    probs_df = extract_amino_acid_probs(pooled, vocab=vocab)
    
    # Filter to residues of interest
    if residues_of_interest:
        valid_indices = [i for i in residues_of_interest.keys() if i < len(probs_df)]
        probs_subset = probs_df.iloc[valid_indices]
        residue_labels = [residues_of_interest[i] for i in valid_indices]
    else:
        probs_subset = probs_df
        residue_labels = [str(i) for i in range(len(probs_df))]
    
    # Scale for visualization
    scaled = scale_logits(probs_subset, method=scale_method)
    
    return {
        "logits_pooled": pooled,
        "probs": probs_subset,
        "scaled_logits": scaled,
        "residue_labels": residue_labels,
        "vocab": vocab,
    }


def save_analysis(
    analysis: Dict,
    output_path: str
) -> None:
    """
    Save logits analysis results.
    
    Args:
        analysis: Output from analyze_residues()
        output_path: Path to save (.csv for probabilities, .pt for full results)
        
    Example:
        >>> save_analysis(analysis, "logits_analysis.csv")
    """
    if output_path.endswith(".csv"):
        df = analysis["probs"].copy()
        df["residue"] = analysis["residue_labels"]
        df = df.set_index("residue")
        df.to_csv(output_path)
    else:
        # Save as PyTorch file
        save_dict = {
            "probs": analysis["probs"].values,
            "scaled_logits": analysis["scaled_logits"],
            "residue_labels": analysis["residue_labels"],
            "vocab": analysis["vocab"],
        }
        torch.save(save_dict, output_path)
