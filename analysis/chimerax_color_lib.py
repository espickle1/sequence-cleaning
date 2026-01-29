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
        method: One of 'quantile', 'power', 'standard', 'robust', or
                'none' (returns values unchanged).

    Returns:
        Scaled 1-D numpy array.
    """
    col = values.reshape(-1, 1)

    if method == "quantile":
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
        method: One of 'quantile', 'power', 'standard', 'robust', or 'none'.

    Returns:
        Fitted scaler object, or None if method is 'none'.
    """
    if method == "none":
        return None

    col = combined_values.reshape(-1, 1)

    if method == "quantile":
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
) -> list[str]:
    """Return ChimeraX ``color`` command lines for each residue."""
    spec = f"#{model}/{chain}" if chain else f"#{model}"
    lines: list[str] = []
    for idx, val in enumerate(normalized):
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
) -> list[str]:
    """Return ChimeraX ``transparency`` command lines for each residue."""
    spec = f"#{model}/{chain}" if chain else f"#{model}"
    lines: list[str] = []
    for idx, val in enumerate(normalized):
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
) -> str:
    """
    Generate a ChimeraX .cxc script string from per-residue values.

    Args:
        values: 1-D array-like of per-residue scalar values (e.g. entropy).
        cmap_name: Seaborn / matplotlib colormap name.
        transform_method: Scaling before min-max normalization.
            One of 'quantile', 'power', 'standard', 'robust', or 'none'.
        color: Whether to include color mapping lines.
        color_invert: Invert the colormap direction.
        transparency: Whether to include transparency mapping lines.
        transparency_invert: Invert the transparency direction.
        model: ChimeraX model number (default 1).
        chain: ChimeraX chain ID (e.g. 'A'). Empty string omits chain.
        show_line: Prepend a ``show`` command at the top.
        targets: ChimeraX target specifiers.

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
            _color_lines(normalized, cmap, model=model, chain=chain, invert=color_invert, targets=targets)
        )

    if transparency:
        lines.extend(
            _transparency_lines(normalized, model=model, chain=chain, invert=transparency_invert, targets=targets)
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
            _color_lines(normalized, cmap, model=model, chain=chain, invert=color_invert, targets=targets)
        )

    if transparency:
        lines.extend(
            _transparency_lines(normalized, model=model, chain=chain, invert=transparency_invert, targets=targets)
        )

    return "\n".join(lines) + "\n"


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
