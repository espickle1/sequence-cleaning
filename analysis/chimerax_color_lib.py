"""
ChimeraX Color Script Generator

Generate ChimeraX .cxc scripts that map per-residue scalar values
(e.g. entropy, conservation scores) to colors and transparency.

Usage:
    from analysis.chimerax_color_lib import generate_chimerax_script

    import numpy as np
    values = np.random.rand(100)  # per-residue values
    script = generate_chimerax_script(values, cmap_name="Greys")
    # Write to file
    write_chimerax_script(script, "output.cxc")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import seaborn as sns
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


# =============================================================================
# SCALING
# =============================================================================

def scale_values(
    values: np.ndarray,
    method: str = "none",
) -> np.ndarray:
    """
    Scale/transform a 1-D array of per-residue values.

    Args:
        values: 1-D array of raw values.
        method: One of 'log', 'minmax', 'quantile', 'power', 'standard', 'robust',
                or 'none' (returns values unchanged).

    Returns:
        Scaled 1-D numpy array.
    """
    col = values.reshape(-1, 1)

    if method == "log":
        # Use log1p (log(1+x)) to safely handle zero values
        out = np.log1p(col)
    elif method == "minmax":
        out = MinMaxScaler().fit_transform(col)
    elif method == "quantile":
        out = QuantileTransformer(output_distribution="uniform").fit_transform(col)
    elif method == "power":
        out = PowerTransformer(method="yeo-johnson").fit_transform(col)
    elif method == "standard":
        out = StandardScaler().fit_transform(col)
    elif method == "robust":
        out = RobustScaler().fit_transform(col)
    else:
        out = col

    return out.flatten()


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize values to the [0, 1] range via MinMaxScaler."""
    return MinMaxScaler().fit_transform(values.reshape(-1, 1)).flatten()


# =============================================================================
# MULTI-FILE SCALING (fit on combined data, apply to individual files)
# =============================================================================

def fit_scaler(
    combined_values: np.ndarray,
    method: str = "none",
):
    """
    Fit a scaler on combined values from multiple files.

    Args:
        combined_values: 1-D array of all values concatenated.
        method: One of 'log', 'minmax', 'quantile', 'power', 'standard', 'robust', or 'none'.

    Returns:
        Fitted scaler object, or None if method is 'none'.
    """
    if method == "none":
        return None

    col = combined_values.reshape(-1, 1)

    if method == "log":
        # Log transform doesn't need fitting, but we use FunctionTransformer
        # for consistent API with other scalers
        scaler = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        scaler.fit(col)  # No-op but keeps API consistent
        return scaler
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "quantile":
        scaler = QuantileTransformer(output_distribution="uniform")
    elif method == "power":
        scaler = PowerTransformer(method="yeo-johnson")
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        return None

    scaler.fit(col)
    return scaler


def scale_values_with_scaler(
    values: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Scale values using a pre-fitted scaler.

    Args:
        values: 1-D array of values to scale.
        scaler: Pre-fitted scaler from fit_scaler(), or None for no transform.

    Returns:
        Scaled 1-D numpy array.
    """
    if scaler is None:
        return values
    col = values.reshape(-1, 1)
    return scaler.transform(col).flatten()


def fit_minmax_scaler(combined_values: np.ndarray) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on combined values from multiple files.

    Args:
        combined_values: 1-D array of all (possibly pre-scaled) values.

    Returns:
        Fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(combined_values.reshape(-1, 1))
    return scaler


def minmax_normalize_with_scaler(
    values: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Normalize values to [0, 1] using a pre-fitted MinMaxScaler.

    Args:
        values: 1-D array of values to normalize.
        scaler: Pre-fitted MinMaxScaler from fit_minmax_scaler().

    Returns:
        Normalized 1-D numpy array clipped to [0, 1].
    """
    normalized = scaler.transform(values.reshape(-1, 1)).flatten()
    # Clip in case individual file has values outside the fitted range
    return np.clip(normalized, 0.0, 1.0)


# =============================================================================
# CXC LINE GENERATORS
# =============================================================================

def _color_lines(
    normalized: np.ndarray,
    cmap,
    model: int = 1,
    chain: str = "",
    invert: bool = False,
    targets: str = "atoms,cartoons,surface",
    residue_mask: np.ndarray | None = None,
) -> list[str]:
    """Return ChimeraX ``color`` command lines for each residue.

    Args:
        normalized: Normalized values [0, 1] for each residue.
        cmap: Colormap object.
        model: ChimeraX model number.
        chain: ChimeraX chain ID.
        invert: Invert colormap direction.
        targets: ChimeraX target specifiers.
        residue_mask: Optional boolean array. If provided, only residues where
            mask is True will have color commands generated.

    Returns:
        List of ChimeraX color command strings.
    """
    spec = f"#{model}/{chain}" if chain else f"#{model}"
    lines: list[str] = []
    for idx, val in enumerate(normalized):
        if residue_mask is not None and not residue_mask[idx]:
            continue
        rgba = cmap(1 - val if invert else val)
        r, g, b, a = (int(round(c * 255)) for c in rgba)
        r, g, b, a = max(r, 0), max(g, 0), max(b, 0), max(a, 0)
        lines.append(f"color {spec}:{idx + 1} {r},{g},{b},{a} {targets}")
    return lines


def _transparency_lines(
    normalized: np.ndarray,
    model: int = 1,
    chain: str = "",
    invert: bool = False,
    targets: str = "atoms,cartoons,surface",
    residue_mask: np.ndarray | None = None,
) -> list[str]:
    """Return ChimeraX ``transparency`` command lines for each residue.

    Args:
        normalized: Normalized values [0, 1] for each residue.
        model: ChimeraX model number.
        chain: ChimeraX chain ID.
        invert: Invert transparency direction.
        targets: ChimeraX target specifiers.
        residue_mask: Optional boolean array. If provided, only residues where
            mask is True will have transparency commands generated.

    Returns:
        List of ChimeraX transparency command strings.
    """
    spec = f"#{model}/{chain}" if chain else f"#{model}"
    lines: list[str] = []
    for idx, val in enumerate(normalized):
        if residue_mask is not None and not residue_mask[idx]:
            continue
        t = max(0.0, min(1.0, 1 - val if invert else val))
        lines.append(f"transparency {spec}:{idx + 1} {t} {targets}")
    return lines


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_chimerax_script(
    values: np.ndarray | Sequence[float],
    *,
    cmap_name: str = "Greys",
    transform_method: str = "none",
    color: bool = True,
    color_invert: bool = False,
    transparency: bool = False,
    transparency_invert: bool = False,
    model: int = 1,
    chain: str = "",
    show_line: bool = True,
    targets: str = "atoms,cartoons,surface",
    residue_mask: np.ndarray | None = None,
) -> str:
    """
    Generate a ChimeraX .cxc script string from per-residue values.

    Args:
        values: 1-D array-like of per-residue scalar values (e.g. entropy).
        cmap_name: Seaborn / matplotlib colormap name.
        transform_method: Scaling before min-max normalization.
            One of 'log', 'minmax', 'quantile', 'power', 'standard', 'robust', or 'none'.
        color: Whether to include color mapping lines.
        color_invert: Invert the colormap direction.
        transparency: Whether to include transparency mapping lines.
        transparency_invert: Invert the transparency direction.
        model: ChimeraX model number (default 1).
        chain: ChimeraX chain ID (e.g. 'A'). Empty string omits chain.
        show_line: Prepend a ``show`` command at the top.
        targets: ChimeraX target specifiers.
        residue_mask: Optional boolean array. If provided, only residues where
            mask is True will have color/transparency commands generated.
            Length must match values array.

    Returns:
        Complete .cxc script as a string.
    """
    values = np.asarray(values, dtype=float)

    # Scale then normalize to [0, 1]
    scaled = scale_values(values, method=transform_method)
    normalized = minmax_normalize(scaled)

    cmap = sns.color_palette(cmap_name, as_cmap=True)

    spec = f"#{model}/{chain}" if chain else f"#{model}"
    lines: list[str] = []

    if show_line:
        lines.append(f"show {spec} surface, atoms, cartoons")

    if color:
        lines.extend(
            _color_lines(
                normalized, cmap, model=model, chain=chain, invert=color_invert,
                targets=targets, residue_mask=residue_mask
            )
        )

    if transparency:
        lines.extend(
            _transparency_lines(
                normalized, model=model, chain=chain, invert=transparency_invert,
                targets=targets, residue_mask=residue_mask
            )
        )

    return "\n".join(lines) + "\n"


def generate_chimerax_script_with_scalers(
    values: np.ndarray | Sequence[float],
    *,
    transform_scaler=None,
    minmax_scaler: MinMaxScaler,
    cmap_name: str = "Greys",
    color: bool = True,
    color_invert: bool = False,
    transparency: bool = False,
    transparency_invert: bool = False,
    model: int = 1,
    chain: str = "",
    show_line: bool = True,
    targets: str = "atoms,cartoons,surface",
    residue_mask: np.ndarray | None = None,
) -> str:
    """
    Generate a ChimeraX .cxc script using pre-fitted scalers.

    Use this when normalizing across multiple entropy files so all sequences
    share the same color scale.

    Args:
        values: 1-D array-like of per-residue scalar values.
        transform_scaler: Pre-fitted scaler from fit_scaler(), or None.
        minmax_scaler: Pre-fitted MinMaxScaler from fit_minmax_scaler().
        cmap_name: Seaborn / matplotlib colormap name.
        color: Whether to include color mapping lines.
        color_invert: Invert the colormap direction.
        transparency: Whether to include transparency mapping lines.
        transparency_invert: Invert the transparency direction.
        model: ChimeraX model number (default 1).
        chain: ChimeraX chain ID (e.g. 'A'). Empty string omits chain.
        show_line: Prepend a ``show`` command at the top.
        targets: ChimeraX target specifiers.
        residue_mask: Optional boolean array. If provided, only residues where
            mask is True will have color/transparency commands generated.
            Length must match values array.

    Returns:
        Complete .cxc script as a string.
    """
    values = np.asarray(values, dtype=float)

    # Scale using pre-fitted scalers
    scaled = scale_values_with_scaler(values, transform_scaler)
    normalized = minmax_normalize_with_scaler(scaled, minmax_scaler)

    cmap = sns.color_palette(cmap_name, as_cmap=True)

    spec = f"#{model}/{chain}" if chain else f"#{model}"
    lines: list[str] = []

    if show_line:
        lines.append(f"show {spec} surface, atoms, cartoons")

    if color:
        lines.extend(
            _color_lines(
                normalized, cmap, model=model, chain=chain, invert=color_invert,
                targets=targets, residue_mask=residue_mask
            )
        )

    if transparency:
        lines.extend(
            _transparency_lines(
                normalized, model=model, chain=chain, invert=transparency_invert,
                targets=targets, residue_mask=residue_mask
            )
        )

    return "\n".join(lines) + "\n"


# =============================================================================
# VALUE FILTERING AND STATISTICS
# =============================================================================

def compute_value_statistics(
    values: np.ndarray,
    transform_method: str = "none",
    transform_scaler=None,
) -> dict:
    """
    Compute statistics of values after applying a transform.

    For quantile transforms, values become ranks (0-1, ordinal).
    For other transforms, values are continuous (rational numbers).

    Args:
        values: 1-D array of raw values.
        transform_method: The transform method used ('quantile' or other).
        transform_scaler: Pre-fitted scaler, or None to fit fresh.

    Returns:
        Dictionary with:
            - is_rank_based: bool - True if quantile (ordinal ranks)
            - transformed_values: np.ndarray - The transformed values
            - min, max, mean, std: float - Statistics of transformed values
            - percentiles: dict - {25, 50, 75} percentile values
    """
    values = np.asarray(values, dtype=float)

    # Apply transform
    if transform_scaler is not None:
        transformed = scale_values_with_scaler(values, transform_scaler)
    else:
        transformed = scale_values(values, method=transform_method)

    is_rank_based = transform_method == "quantile"

    stats = {
        "is_rank_based": is_rank_based,
        "transformed_values": transformed,
        "min": float(np.min(transformed)),
        "max": float(np.max(transformed)),
        "mean": float(np.mean(transformed)),
        "std": float(np.std(transformed)),
        "percentiles": {
            25: float(np.percentile(transformed, 25)),
            50: float(np.percentile(transformed, 50)),
            75: float(np.percentile(transformed, 75)),
        },
        "n_residues": len(transformed),
    }
    return stats


def parse_range_string(range_str: str) -> list[tuple[float, float]]:
    """
    Parse a string specifying one or more value ranges.

    Supports formats like:
        "0-0.25"              -> [(0, 0.25)]
        "0-0.25, 0.75-1.0"    -> [(0, 0.25), (0.75, 1.0)]
        "0.1 - 0.3, 0.6-0.9"  -> [(0.1, 0.3), (0.6, 0.9)]

    Args:
        range_str: Comma-separated ranges, each as "min-max".

    Returns:
        List of (min, max) tuples.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    if not range_str or range_str.strip().lower() in ("all", "none", ""):
        return []

    ranges = []
    parts = range_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Split on hyphen, handling negative numbers
        # Use a regex to handle cases like "-0.5-0.5" or "0.5--0.5"
        import re
        match = re.match(r"^\s*(-?\d*\.?\d+)\s*-\s*(-?\d*\.?\d+)\s*$", part)
        if not match:
            raise ValueError(f"Invalid range format: '{part}'. Expected 'min-max'.")
        lo, hi = float(match.group(1)), float(match.group(2))
        if lo > hi:
            lo, hi = hi, lo  # Swap if reversed
        ranges.append((lo, hi))
    return ranges


def create_value_mask(
    values: np.ndarray,
    ranges: list[tuple[float, float]],
    inclusive: bool = True,
) -> np.ndarray:
    """
    Create a boolean mask indicating which values fall within specified ranges.

    Args:
        values: 1-D array of values to filter.
        ranges: List of (min, max) tuples. Values within ANY range are included.
        inclusive: If True, endpoints are included (<=, >=). If False, strict (<, >).

    Returns:
        Boolean array of same shape as values. True = include this residue.
    """
    if not ranges:
        # No filtering - include all
        return np.ones(len(values), dtype=bool)

    mask = np.zeros(len(values), dtype=bool)
    for lo, hi in ranges:
        if inclusive:
            mask |= (values >= lo) & (values <= hi)
        else:
            mask |= (values > lo) & (values < hi)
    return mask


def display_statistics_summary(
    stats: dict,
    label: str = "",
) -> str:
    """
    Format statistics for display to the user.

    Args:
        stats: Dictionary from compute_value_statistics().
        label: Optional label for the sequence/file.

    Returns:
        Formatted string for printing.
    """
    lines = []
    if label:
        lines.append(f"=== Statistics for {label} ===")
    else:
        lines.append("=== Transformed Value Statistics ===")

    if stats["is_rank_based"]:
        lines.append("Transform type: Quantile (rank-based, ordinal values 0-1)")
        lines.append(f"  Residues: {stats['n_residues']}")
        lines.append(f"  Rank range: {stats['min']:.4f} - {stats['max']:.4f}")
        lines.append("  Quartiles:")
        lines.append(f"    Q1 (25%): {stats['percentiles'][25]:.4f}")
        lines.append(f"    Q2 (50%): {stats['percentiles'][50]:.4f}")
        lines.append(f"    Q3 (75%): {stats['percentiles'][75]:.4f}")
        lines.append("\n  Suggested rank ranges:")
        lines.append("    Bottom quartile: 0-0.25")
        lines.append("    Middle 50%: 0.25-0.75")
        lines.append("    Top quartile: 0.75-1.0")
    else:
        lines.append(f"Transform type: {stats.get('transform_method', 'numeric')} (continuous values)")
        lines.append(f"  Residues: {stats['n_residues']}")
        lines.append(f"  Min: {stats['min']:.6f}")
        lines.append(f"  Max: {stats['max']:.6f}")
        lines.append(f"  Mean: {stats['mean']:.6f}")
        lines.append(f"  Std: {stats['std']:.6f}")
        lines.append("  Percentiles:")
        lines.append(f"    25%: {stats['percentiles'][25]:.6f}")
        lines.append(f"    50%: {stats['percentiles'][50]:.6f}")
        lines.append(f"    75%: {stats['percentiles'][75]:.6f}")

    return "\n".join(lines)


# =============================================================================
# FILE I/O
# =============================================================================

def write_chimerax_script(
    script: str,
    output_path: str | Path,
) -> Path:
    """
    Write a .cxc script string to a file.

    Args:
        script: The script content (from ``generate_chimerax_script``).
        output_path: Destination file path.

    Returns:
        Resolved Path of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    return path.resolve()
