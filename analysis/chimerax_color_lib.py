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
# CXC LINE GENERATORS
# =============================================================================

def _color_lines(
    normalized: np.ndarray,
    cmap,
    model: int = 1,
    invert: bool = False,
    targets: str = "atoms,cartoons,surface",
) -> list[str]:
    """Return ChimeraX ``color`` command lines for each residue."""
    lines: list[str] = []
    for idx, val in enumerate(normalized):
        rgba = cmap(1 - val if invert else val)
        r, g, b, a = (int(round(c * 255)) for c in rgba)
        r, g, b, a = max(r, 0), max(g, 0), max(b, 0), max(a, 0)
        lines.append(f"color #{model}:{idx + 1} {r},{g},{b},{a} {targets}")
    return lines


def _transparency_lines(
    normalized: np.ndarray,
    model: int = 1,
    invert: bool = False,
    targets: str = "atoms,cartoons,surface",
) -> list[str]:
    """Return ChimeraX ``transparency`` command lines for each residue."""
    lines: list[str] = []
    for idx, val in enumerate(normalized):
        t = max(0.0, min(1.0, 1 - val if invert else val))
        lines.append(f"transparency #{model}:{idx + 1} {t} {targets}")
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

    lines: list[str] = []

    if show_line:
        lines.append(f"show #{model} surface, atoms, cartoons")

    if color:
        lines.extend(
            _color_lines(normalized, cmap, model=model, invert=color_invert, targets=targets)
        )

    if transparency:
        lines.extend(
            _transparency_lines(normalized, model=model, invert=transparency_invert, targets=targets)
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
