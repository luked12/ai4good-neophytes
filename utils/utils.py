"""
===============================================================================
Small shared helpers (paths, dict/array juggling, image denormalization)
===============================================================================
"""

import os
import re

import numpy as np
from omegaconf import ListConfig, DictConfig


# =============================================================================
def denormalize(image, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    """CHW tensor-style array -> HWC image in [0, 1] (undoes Normalize)."""
    image = image.transpose((1, 2, 0))
    image = (image * std + mean)
    return np.clip(image, 0, 1)


# =============================================================================
def getListOfFiles(dirName):
    """All files below dirName, recursively."""
    allFiles = []
    for entry in os.listdir(dirName):
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles += getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


# =============================================================================
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Sort key for human ordering: epoch=9 before epoch=10.

    https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# =============================================================================
def make_dir(directory, name):
    """Create <directory>/<name> if missing and return its path."""
    folder = os.path.join(directory, name)
    os.makedirs(folder, exist_ok=True)
    return folder


# =============================================================================
def convert_ndarray_to_list(data):
    """Recursively turn numpy types into JSON-serializable Python types."""
    if isinstance(data, dict):
        return {convert_ndarray_to_list(k): convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    return data


# =============================================================================
def shrink_dict(original_dict, keep_keys):
    """Subset of a dict, keeping only the keys in keep_keys that exist."""
    return {key: original_dict[key] for key in keep_keys if key in original_dict}


# =============================================================================
def extract_dataset_name(file_path, known_dataset_names):
    """First known dataset name found in the path (searched innermost first).

    The class-value mapping in the data config is keyed by dataset name, so each
    tile has to be traced back to the dataset folder it came from.
    """
    for part in reversed(file_path.split(os.sep)):
        if part in known_dataset_names:
            return part
    raise ValueError(f"No known dataset name found in path: {file_path}")


# =============================================================================
def ensure_list_values(dirs_dict):
    """Make every leaf of the image_dir config a list, so a single path works too."""
    for key, value in dirs_dict.items():
        if isinstance(value, (dict, DictConfig)):
            ensure_list_values(value)
        else:
            dirs_dict[key] = value if isinstance(value, (ListConfig, list)) else [value]
    return dirs_dict
