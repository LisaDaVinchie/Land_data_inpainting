import json
from pathlib import Path

def load_config(json_file: Path, categories=None):
    """
    Loads a JSON file and returns a dictionary of the specified categories.
    
    Parameters:
        json_file (str): Path to the JSON file.
        categories (list, optional): List of categories to import. If None, all categories are imported.
    
    Returns:
        dict: A dictionary containing the loaded variables.
    """
    # Load the JSON file
    with open(json_file, 'r') as file:
        config = json.load(file)

    # If categories are specified, only load those
    if categories:
        result = {}
        for category in categories:
            if category in config:
                result[category] = config[category]
        return result
    else:
        # Load all categories if none are specified
        return config