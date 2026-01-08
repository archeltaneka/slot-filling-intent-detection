import os
import yaml


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