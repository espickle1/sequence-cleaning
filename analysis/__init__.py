# Analysis Package
# Provides functions for entropy calculation and logits analysis

from .entropy_lib import (
    calculate_entropy,
    calculate_entropy_batched,
    analyze_entropy,
    get_constrained_positions,
    get_flexible_positions,
    entropy_summary,
)

from .logits_lib import (
    pool_logits,
    scale_logits,
    extract_amino_acid_probs,
    plot_heatmap,
    analyze_residues,
)

__all__ = [
    # Entropy analysis
    "calculate_entropy",
    "calculate_entropy_batched",
    "analyze_entropy",
    "get_constrained_positions",
    "get_flexible_positions",
    "entropy_summary",
    # Logits analysis
    "pool_logits",
    "scale_logits",
    "extract_amino_acid_probs",
    "plot_heatmap",
    "analyze_residues",
]
