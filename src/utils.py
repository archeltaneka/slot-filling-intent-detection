import os
import yaml
import torch


def load_config_file(file_path):
    """
    Load and parse a YAML config file safely.
    Returns the parsed data as a Python object (dict, list, etc.).
    """
    # Validate file existence
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Use safe_load to avoid executing arbitrary code
            data = yaml.safe_load(file)
            if data is None:
                return {}  # Return empty dict if YAML is empty
            return data
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")


def save_model(model, save_dir, model_name):
    """
    Save a PyTorch model to a file in the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_path)