import math
import os
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def resolve_path(path, base_dir=None):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        base_dir = Path.cwd()
    return Path(base_dir) / path


def get_matching_halfna_name(fullna_filename):
    stem, ext = os.path.splitext(fullna_filename)
    return f"{stem}_halfNa{ext}"


def chunk_indices(n_items, n_chunks, chunk_id):
    chunk_size = math.ceil(n_items / n_chunks)
    start = chunk_id * chunk_size
    end = min(start + chunk_size, n_items)
    return start, end


def slurm_array_info(default_jobs=1):
    n_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", str(default_jobs)))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    return n_jobs, task_id
