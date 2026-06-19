from pathlib import Path


def list_cif_files(directory):
    directory = Path(directory)
    return sorted([p for p in directory.iterdir() if p.suffix.lower() == ".cif"])
