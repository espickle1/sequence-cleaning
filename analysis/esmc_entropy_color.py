# %%
## Import necessary libraries
from __future__ import annotations

# Standard libraries
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Any

# Specialized libraries
import seaborn as sns
from sklearn.preprocessing import (
    MinMaxScaler, 
    QuantileTransformer, 
    PowerTransformer, 
    StandardScaler, 
    RobustScaler
)

# %%
## Create class containing read and combine configuration
class ColorConfig:
    def __init__(self, config_path: str | Path):
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        # Loading paths
        self.input_dir = Path(self._cfg["paths"]["input_dir"])
        self.output_dir = Path(self._cfg["paths"]["output_dir"])

    @staticmethod
    # Get the original (resolved) YAML as a list
    def _as_list(x: Any) -> list[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [t.strip() for t in str(x).split(",") if t.strip()]

    # Or get the original (resolved) YAML as a dict
    def to_dict(self) -> dict:
        return self._cfg

# %%
## Calculate the rgb color based on the input scalar
def generate_palette_rgb(
            cmap,
            data_minmax,
            model_number,
            chain_id: str,
            color_invert: str,
            targets: str = "atoms,cartoons,surface",
            ):
      spec = f"#{model_number}/{chain_id}" if chain_id else f"#{model_number}"
      color_list = []

      # Calculate the color mapping
      for idx, value in enumerate(data_minmax):
            # Correct if the colormap is inverted
            if color_invert == "True":
                rgba_0_1 = cmap(1 - value)
            else:
                rgba_0_1 = cmap(value)

            # Calculate the RGB color mapping
            rgb_255 = tuple(int(round(c * 255)) for c in rgba_0_1)
            rgb_255 = [max(color_value, 0) for color_value in rgb_255]
            color_text = f"color {spec}:{idx + 1} {rgb_255[0]},{rgb_255[1]},{rgb_255[2]},{rgb_255[3]} {targets}"
            color_list.append(str(color_text))

      return color_list

# %%
## Calculate transparency based on the input scalar
def generate_transparency(
            data_minmax,
            model_number,
            chain_id: str,
            transparency_invert: str,
            targets: str = "atoms,cartoons,surface",
            ):
      spec = f"#{model_number}/{chain_id}" if chain_id else f"#{model_number}"
      transparency_list = []

      # Calculate the transparency mapping
      for idx, value in enumerate(data_minmax):
            # Correct if the transparency is inverted
            if transparency_invert == "True":
                transparency = 1 - value
            else:
                transparency = value

            # Calculate the transparency mapping
            transparency = max(0, min(1, transparency))
            transparency_text = f"transparency {spec}:{idx + 1} {transparency} {targets}"
            transparency_list.append(str(transparency_text))

      return transparency_list

# %%
## Scale and normalize the data
def data_scaler(entropy_mean_values, transform_method: str):
    """
    Apply various scaling/transformation methods to entropy values.
    
    Args:
        entropy_mean_values: Input data to be scaled
        transform_method: Type of transformation ('quantile', 'power', 'standard', 'robust', or other)
    
    Returns:
        Scaled/transformed values as a flattened numpy array
    """
    if transform_method == "quantile":
        # Quantile transformer maps values to uniform distribution [0, 1]
        scaled_values = QuantileTransformer(output_distribution="uniform").fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    elif transform_method == "power":
        # Power transformer applies Yeo-Johnson transformation for skewed data
        scaled_values = PowerTransformer(method="yeo-johnson").fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    elif transform_method == "standard":
        # Standard scaler normalizes to zero mean and unit variance
        scaled_values = StandardScaler().fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    elif transform_method == "robust":
        # Robust scaler uses median and IQR, less sensitive to outliers
        scaled_values = RobustScaler().fit_transform(
            entropy_mean_values.values.reshape(-1, 1)
        ).flatten()
    else:
        # Default: return values as-is without transformation
        scaled_values = entropy_mean_values.values.flatten()
        
    return scaled_values

# %%
## Prompt the user for runtime parameters
def prompt_user_parameters() -> dict:
    """
    Interactively prompt the user for colormap, transform, color/transparency
    mode, and model/chain ID settings.

    Returns:
        Dictionary of user-selected parameters.
    """
    print("\n=== ChimeraX Color Script Configuration ===\n")

    # --- Entropy files ---
    print("Enter entropy CSV filenames (relative to input_dir), separated by commas.")
    raw = input("Entropy file(s): ").strip()
    input_files = [f.strip() for f in raw.split(",") if f.strip()]
    if not input_files:
        print("Error: at least one entropy file is required.")
        raise SystemExit(1)

    # --- Colormap ---
    cmap_name = input("Colormap name [Greys]: ").strip() or "Greys"

    # --- Transform method ---
    print("Transform methods: none, quantile, power, standard, robust")
    transform_method = input("Transform method [none]: ").strip() or "none"

    # --- Color / color invert / transparency mode ---
    color_mapping = input("Enable color mapping? (y/n) [y]: ").strip().lower()
    color_mapping = color_mapping not in ("n", "no")

    color_invert = False
    if color_mapping:
        ci = input("Invert colormap? (y/n) [n]: ").strip().lower()
        color_invert = ci in ("y", "yes")

    transparency_mapping = input("Enable transparency mapping? (y/n) [n]: ").strip().lower()
    transparency_mapping = transparency_mapping in ("y", "yes")

    transparency_invert = False
    if transparency_mapping:
        ti = input("Invert transparency? (y/n) [n]: ").strip().lower()
        transparency_invert = ti in ("y", "yes")

    # --- Model and chain ID ---
    model_str = input("Model ID [1]: ").strip() or "1"
    model_number = int(model_str)

    chain_id = input("Chain ID (e.g. A, B — leave blank for none): ").strip()

    return {
        "input_files": input_files,
        "cmap_name": cmap_name,
        "transform_method": transform_method,
        "color_mapping": color_mapping,
        "color_invert": color_invert,
        "transparency_mapping": transparency_mapping,
        "transparency_invert": transparency_invert,
        "model_number": model_number,
        "chain_id": chain_id,
    }


## Generate the color map
def generate_color_map(config_path: str):
    """
    Generate color and transparency mappings for molecular visualization.

    Args:
        config_path: Path to the YAML configuration file containing input/output paths.
    """
    # Load configuration from YAML file (paths only)
    cfg = ColorConfig(config_path)

    # Prompt user for runtime parameters
    params = prompt_user_parameters()
    input_files = params["input_files"]
    cmap_name = params["cmap_name"]
    transform_method = params["transform_method"]
    color_mapping = params["color_mapping"]
    color_invert = params["color_invert"]
    transparency_mapping = params["transparency_mapping"]
    transparency_invert = params["transparency_invert"]
    model_number = params["model_number"]
    chain_id = params["chain_id"]

    # String versions for generate_palette_rgb / generate_transparency
    color_invert_str = "True" if color_invert else "False"
    transparency_invert_str = "True" if transparency_invert else "False"

    # Construct colormap portion of output filename
    cmap_name_file = cmap_name
    if color_invert:
        cmap_name_file = cmap_name_file + "_invert"

    cxc_file_prefix = f"entropy_color_mapping_{cmap_name_file}"
    if transparency_mapping:
        cxc_file_prefix = cxc_file_prefix + "_transparency"

    # Load the colormap from seaborn (shared across files)
    cmap = sns.color_palette(cmap_name, as_cmap=True)

    # Process each entropy file
    for filename in input_files:
        input_path = cfg.input_dir / filename
        file_stem = Path(input_path).stem

        print(f"\nProcessing: {input_path}")

        # Read entropy mean values from CSV file
        entropy_mean = pd.read_csv(input_path)

        # Scale and normalize entropy values with transformer
        entropy_transformed = data_scaler(entropy_mean, transform_method)

        # Normalize entropy values to 0-1 range using MinMaxScaler
        data_minmax = MinMaxScaler().fit_transform(
            entropy_transformed.reshape(-1, 1)).flatten()

        # Initialize lists for this file
        color_list_total = []
        transparency_list_total = []

        # Generate color mappings if enabled
        if color_mapping:
            color_list = generate_palette_rgb(
                cmap=cmap,
                data_minmax=data_minmax,
                model_number=model_number,
                chain_id=chain_id,
                color_invert=color_invert_str,
            )
            color_list_total.extend(color_list)

        # Generate transparency mappings if enabled
        if transparency_mapping:
            transparency_list = generate_transparency(
                data_minmax=data_minmax,
                model_number=model_number,
                chain_id=chain_id,
                transparency_invert=transparency_invert_str,
            )
            transparency_list_total.extend(transparency_list)

        # Write color and transparency mappings to output CXC file
        output_file = cfg.output_dir / f"{cxc_file_prefix}_{file_stem}.cxc"
        with open(output_file, "w") as f:
            if color_mapping:
                for residue in color_list_total:
                    f.write(f"{residue}\n")
            if transparency_mapping:
                for residue in transparency_list_total:
                    f.write(f"{residue}\n")
            else:
                f.write("select clear")

        print(f"  -> {output_file}")

    print(f"\nDone. {len(input_files)} file(s) processed.")
    return

# %%
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python esmc_entropy_color.py <config.yaml>")
        sys.exit(1)
    generate_color_map(sys.argv[1])
