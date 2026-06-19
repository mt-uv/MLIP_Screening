import argparse
from pathlib import Path

import pandas as pd

from mlip_screening.calculators import build_calculator
from mlip_screening.io import list_cif_files
from mlip_screening.relax import safe_relax_structure
from mlip_screening.utils import chunk_indices, get_matching_halfna_name, load_config, slurm_array_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    model = config["model"].lower()
    fullna_dir = Path(config["fullna_dir"]).expanduser()
    halfna_dir = Path(config["halfna_dir"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    fmax = float(config.get("fmax", 0.02))
    max_steps = int(config.get("max_steps", 500))
    default_jobs = int(config.get("default_array_jobs", 1))

    n_jobs, task_id = slurm_array_info(default_jobs=default_jobs)

    fullna_files = list_cif_files(fullna_dir)
    if not fullna_files:
        raise RuntimeError(f"No CIF files found in {fullna_dir}")

    start, end = chunk_indices(len(fullna_files), n_jobs, task_id)
    assigned_files = fullna_files[start:end]

    print(f"Model: {model}")
    print(f"Array task {task_id} / {n_jobs}")
    print(f"Total fullNa files: {len(fullna_files)}")
    print(f"Task range: [{start}:{end}), count = {len(assigned_files)}")

    columns = [
        "model",
        "composition",
        "relaxed_energy_eV",
        "relaxed_volume_A3",
        "relaxation_time_s",
        "halfNa_composition",
        "halfNa_relaxed_energy_eV",
        "halfNa_relaxed_volume_A3",
        "halfNa_relaxation_time_s",
        "full_converged",
        "halfNa_converged",
        "pair_converged",
        "full_status",
        "full_error",
        "halfNa_status",
        "halfNa_error",
    ]

    out_csv = output_dir / f"{model}_results_task_{task_id:02d}.csv"
    out_xlsx = output_dir / f"{model}_results_task_{task_id:02d}.xlsx"

    if not assigned_files:
        pd.DataFrame(columns=columns).to_csv(out_csv, index=False)
        pd.DataFrame(columns=columns).to_excel(out_xlsx, index=False)
        return

    calculator = build_calculator(config)
    rows = []

    for idx, fullna_path in enumerate(assigned_files, start=1):
        fullna_file = fullna_path.name
        fullna_stem = fullna_path.stem
        halfna_file = get_matching_halfna_name(fullna_file)
        halfna_path = halfna_dir / halfna_file
        halfna_stem = Path(halfna_file).stem

        print(f"[{idx}/{len(assigned_files)}] {fullna_file}")

        e_full, v_full, t_full, conv_full, status_full, err_full = safe_relax_structure(
            fullna_path, calculator, fmax=fmax, steps=max_steps
        )

        if halfna_path.exists():
            e_half, v_half, t_half, conv_half, status_half, err_half = safe_relax_structure(
                halfna_path, calculator, fmax=fmax, steps=max_steps
            )
        else:
            e_half, v_half, t_half, conv_half = None, None, None, False
            status_half, err_half = "missing", "Matching _halfNa.cif not found"

        rows.append({
            "model": model,
            "composition": fullna_stem,
            "relaxed_energy_eV": e_full,
            "relaxed_volume_A3": v_full,
            "relaxation_time_s": t_full,
            "halfNa_composition": halfna_stem,
            "halfNa_relaxed_energy_eV": e_half,
            "halfNa_relaxed_volume_A3": v_half,
            "halfNa_relaxation_time_s": t_half,
            "full_converged": conv_full,
            "halfNa_converged": conv_half,
            "pair_converged": bool(conv_full and conv_half),
            "full_status": status_full,
            "full_error": err_full,
            "halfNa_status": status_half,
            "halfNa_error": err_half,
        })

        if config.get("write_partial", True):
            pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)
            pd.DataFrame(rows, columns=columns).to_excel(out_xlsx, index=False)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved Excel: {out_xlsx}")


if __name__ == "__main__":
    main()
