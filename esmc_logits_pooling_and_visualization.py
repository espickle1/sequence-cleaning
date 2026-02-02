# %%
# Import required libraries for data processing, visualization, and ML operations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

import torch

from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer, StandardScaler

# %%
# Add project tools to system path for custom module imports
import sys
tools_path = "/home/azureuser/cloudfiles/code/Users/jc62/projects/esm3/code/"
sys.path.extend([tools_path])

# %%
class AnalysisConfig:
    """
    Configuration class for logits analysis.
    
    Loads and manages configuration parameters from a YAML file including:
    - File paths for data and output
    - PyTorch loading settings
    - Sequence vocabulary definitions
    - Residue positions of interest
    - Scaling methodology
    """
    def __init__(self, config_path: str | Path):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # Paths - Define directories and file locations
        self.tools_path = cfg["paths"]["tools_path"]
        self.analysis_directory = Path(cfg["paths"]["analysis_directory"])
        self.logits_file = Path(cfg["paths"]["logits_file"])
        
        # Torch load configuration - Settings for loading PyTorch tensors
        self.torch_config = cfg["torch_load"]
        
        # Sequence vocabulary - Extract amino acid vocabulary subset
        self.sequence_vocab_full = cfg["sequence_vocab"]["full"]
        vocab_start = cfg["sequence_vocab"]["amino_acid_start"]
        vocab_end = cfg["sequence_vocab"]["amino_acid_end"]
        self.sequence_vocab = self.sequence_vocab_full[vocab_start:vocab_end]
        
        # Residues of interest - Convert string keys to integers for indexing
        self.residues_of_interest = {
            int(k): v for k, v in cfg["residues_of_interest"].items()
        }
        
        # Scaler method - Normalization/scaling strategy
        self.scaler_method = cfg["scaler_method"]

        # Specific indices - Subset of logits to analyze
        self.specific_indices = cfg["specific_indices"]
    
    def load_logits(self):
        """
        Load logits data from file using configured parameters.
        
        Returns:
            tuple: (logits tensor, sequence id)
        """
        # Torch loading parameters - Handle device mapping
        map_location = self.torch_config["map_location"]
        if map_location == "cpu":
            map_location = torch.device('cpu')
        
        # Load logits tensor from saved file
        polymerase_logits = torch.load(
            self.logits_file,
            weights_only=self.torch_config["weights_only"],
            map_location=map_location
        )

        return polymerase_logits['logits'], polymerase_logits['id']

# %%
def logits_pooling_and_processing(
        logits, 
        specific_indices: list, 
        sequence_vocab: list, 
        residues_of_interest: dict
        ):
    """
    Pool and process logits data for specified indices and residues.
    
    Performs mean pooling across selected indices and extracts amino acid probabilities
    for residues of interest.
    
    Args:
        logits: Full logits tensor from model output
        specific_indices: List of indices to pool over
        sequence_vocab: List of amino acid tokens
        residues_of_interest: Dictionary mapping residue positions to labels
        
    Returns:
        pd.DataFrame: Processed logits for residues of interest with amino acid columns
    """
    # Extract subset of logits for specified indices
    logits_subset = torch.stack([logits[i] for i in specific_indices])

    # Apply mean pooling across the selected indices
    logits_mean_pool_subset = logits_subset.mean(dim=0)

    # Extract amino acid logits (skip first 4 special tokens)
    logits_amino_acids = logits_mean_pool_subset[:, 4:len(sequence_vocab)+4]

    # Convert to DataFrame for easier manipulation
    logits_cpu_df = pd.DataFrame(
        logits_amino_acids.to(torch.float32).cpu().numpy(), 
        columns=sequence_vocab
    )
    
    # Filter to only residues of interest
    logits_residues_of_interest = logits_cpu_df.loc[list(residues_of_interest.keys())]

    return logits_residues_of_interest

# %%
def logits_scaler(logits, method: str):
    """
    Scale logits data using the specified normalization method.
    
    Applies various scaling transformations to normalize logits values
    for better visualization and comparison.

    Args:
        logits: DataFrame containing logits values
        method: Scaling method - 'minmax', 'robust', 'power', or 'standard'

    Returns:
        np.ndarray: Scaled logits with same shape as input
        
    Raises:
        ValueError: If method is not one of the supported options
    """
    original_shape = logits.shape

    # Select scaler based on method
    if method == 'minmax':
        scaler = MinMaxScaler()  # Scale to [0, 1] range
    elif method == 'robust':
        scaler = RobustScaler()  # Scale using median and IQR (robust to outliers)
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)  # Power transform to make data more Gaussian
    elif method == 'standard':
        scaler = StandardScaler()  # Standardize to mean=0, std=1
    else:
        raise ValueError("Invalid method. Choose from 'minmax', 'robust', 'power', 'standard'.")
    
    # Apply scaling (transpose for feature-wise scaling, then transpose back)
    scaled_logits = scaler.fit_transform(logits.values.T).T
    scaled_logits = scaled_logits.reshape(original_shape)

    print(f"Scaling method: {method}")

    return scaled_logits

# %%
def heatmap_generator(
        data_scaled, 
        title: str,
        fig_size: tuple,
        sequence_vocab: list, 
        residues_of_interest: dict, 
        cmap: str, 
        vmin: float, vmax: float
        ):
    """
    Generate a heatmap visualization of scaled logits data.
    
    Creates a color-coded heatmap showing amino acid propensities across
    residues of interest.
    
    Args:
        data_scaled: Scaled logits data (2D array)
        title: Plot title
        fig_size: Figure dimensions (width, height)
        sequence_vocab: List of amino acid labels for x-axis
        residues_of_interest: Dictionary of residue positions and labels for y-axis
        cmap: Matplotlib colormap name
        vmin: Minimum value for color scale (None for auto)
        vmax: Maximum value for color scale (None for auto)
    """
    # Set up the plot with specified dimensions
    fig, ax = plt.subplots(figsize=fig_size)

    # Create heatmap with specified colormap and value range
    im = ax.imshow(
        data_scaled, 
        cmap=cmap, 
        aspect='auto', 
        vmin=vmin, 
        vmax=vmax
    )

    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Variant Propensity', fontsize=9)

    # Customize axis labels and title
    ax.set_xlabel('Amino Acid', fontsize=11)
    ax.set_ylabel('Residues', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set tick positions and labels
    ax.set_xticks(np.arange(len(sequence_vocab)))
    ax.set_xticklabels(sequence_vocab, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(residues_of_interest)))
    ax.set_yticklabels(list(residues_of_interest.values()))

    plt.tight_layout()
    plt.show()

    return

# %%
def logits_loading_pooling_and_plotting(config_path: Path):
    """
    Main pipeline function to load, process, and visualize logits data.
    
    Orchestrates the complete workflow:
    1. Load configuration
    2. Load logits from file
    3. Pool and process logits
    4. Scale the data
    5. Generate visualization
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration parameters
    config = AnalysisConfig(config_path)
    
    # Load logits tensor and sequence IDs
    logits, ids = config.load_logits()

    # Pool logits across specified indices and extract residues of interest
    logits_pooled = logits_pooling_and_processing(
        logits=logits,
        specific_indices=config.specific_indices,
        sequence_vocab=config.sequence_vocab,
        residues_of_interest=config.residues_of_interest
        )    

    # Apply scaling/normalization
    data_scaled = logits_scaler(
        logits_pooled, 
        method=config.scaler_method
        )

    # Generate heatmap visualization
    heatmap_generator(
        data_scaled=data_scaled, 
        title='Amino Acid Likelihood (from Logits) For Residues of Interest',
        fig_size=(7, 3),
        sequence_vocab=config.sequence_vocab,
        residues_of_interest=config.residues_of_interest,
        cmap="coolwarm", 
        vmin=None, 
        vmax=None
    )

    return

# %%
# Main execution block
if __name__ == "__main__":
    # Run the complete analysis pipeline
    # TODO: Provide actual config path
    logits_loading_pooling_and_plotting(
        config_path=Path()
    )
