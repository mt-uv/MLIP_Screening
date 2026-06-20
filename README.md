# MLIP Screening of Layered Sodium Transition-Metal Oxides

This repository contains the workflows used to relax full-Na and half-Na layered sodium transition-metal oxide structures using multiple machine-learning interatomic potentials (MLIPs): UMA, MACE, NequIP, SevenNet, ORB-v3, and EquiformerV3.

## Repository structure

```text
mlip-screening/
├── checkpoints/
├── configs/
├── data/
│   ├── example_fullNa/
│   └── example_halfNa/
├── environments/
├── results/
├── scripts/
├── src/
├── LICENSE
├── pyproject.toml
└── README.md
```

## Input structure convention

Full-Na CIF files should be placed in:

```text
data/example_fullNa/
```

Half-Na CIF files should be placed in:

```text
data/example_halfNa/
```

The half-Na file must use the same base name as the full-Na file with `_halfNa` appended.

Example:

```text
data/example_fullNa/Na12Co1Cr1Fe1Mn9O24.cif
data/example_halfNa/Na12Co1Cr1Fe1Mn9O24_halfNa.cif
```

For full production runs, replace the example folders in the configuration files with the corresponding full dataset folders.

## Dataset availability

Only representative example CIF files are included in this repository.

The complete CIF dataset used in the study is given in figshare

```text
Full dataset: https://figshare.com/s/3f42c88bf81e25332716
```


## Installation

Separate Conda environments are provided for each MLIP in the `environments/` directory.

Example for MACE:

```bash
conda env create -f environments/mace.yml
conda activate mace-screening
pip install -e .
```

Example for ORB-v3:

```bash
conda env create -f environments/orb.yml
conda activate orb-screening
pip install -e .
```

Example for SevenNet:

```bash
conda env create -f environments/sevennet.yml
conda activate sevennet-screening
pip install -e .
```

Example for UMA:

```bash
conda env create -f environments/uma.yml
conda activate uma-screening
pip install -e .
```

## Running relaxations

Each model is controlled by a YAML file in `configs/`.

### MACE

```bash
python scripts/run_relaxations.py --config configs/mace.yaml
```

### UMA

```bash
python scripts/run_relaxations.py --config configs/uma.yaml
```

### SevenNet

```bash
python scripts/run_relaxations.py --config configs/sevennet.yaml
```

### ORB-v3

```bash
python scripts/run_relaxations.py --config configs/orb.yaml
```

### NequIP

NequIP requires a compiled model file. Set the model path before running:

```bash
export NEQUIP_MODEL=/checkpoints/model.nequip.pt2
python scripts/run_relaxations.py --config configs/nequip.yaml
```

If the `.pt2` model was compiled for CUDA, it must be run on a CUDA-enabled machine.

### EquiformerV3

EquiformerV3 calculations were performed using the EquiformerV3 source code.

Clone or place the EquiformerV3 source code separately, 

https://github.com/atomicarchitects/equiformer_v3

then set:

```bash
export EQUIFORMER_V3_ROOT=/path/to/equiformer_v3
export PYTHONPATH=$EQUIFORMER_V3_ROOT/src:$PYTHONPATH
```

Then run:

```bash
python scripts/run_relaxations.py --config configs/equiformer.yaml
```

The EquiformerV3 configuration uses the pretrained checkpoint:

```text
mirror-physics/equiformer_v3
checkpoint/omat24-mptrj-salex_gradient.pt
```

and the original EquiformerV3 configuration file:

```text
experimental/configs/omat24/salex_mptrj/experiments/gradient/equiformer_v3_grad-finetune_N@7_L@4_attn-hidden@32_rbf@64_max-neighbors@300_attn-grid@14-8_ffn-grid@14_attn-eps@1e-8_lr@0-5e-5-warmup@0.1-epochs@2-mptrj-salex-ratio@8-bs@256-wd@1e-3-beta2@0.98-eps@1e-6_pt-reg-dens-ft-no-reg-lr@1e-4.yml
```

EquiformerV3 is recommended to be run on a CUDA-enabled Linux system using the environment provided with the EquiformerV3 source code.

## SLURM array execution

The workflow supports SLURM array jobs. Example:

```bash
sbatch scripts/submit_slurm_array.sh configs/mace.yaml
```

The number of array jobs is read from the SLURM environment. Each task processes a subset of the CIF files and writes separate output files.

## Output files

Each run writes CSV and XLSX files to the corresponding model folder in `results/`.

Example:

```text
results/mace/mace_results_task_00.csv
results/mace/mace_results_task_00.xlsx
```

The output contains:

```text
composition
relaxed_energy_eV
relaxed_volume_A3
relaxation_time_s
halfNa_composition
halfNa_relaxed_energy_eV
halfNa_relaxed_volume_A3
halfNa_relaxation_time_s
full_converged
halfNa_converged
pair_converged
full_status
full_error
halfNa_status
halfNa_error
```

## Merging array results

After all SLURM array jobs finish, merge the outputs:

```bash
python scripts/merge_results.py \
    --results-dir results/mace \
    --prefix mace_results_task \
    --output results/mace/mace_results_merged.csv
```

Change the model folder and prefix as needed for UMA, NequIP, SevenNet, ORB-v3, or EquiformerV3.

## Notes

* The example CIF files are provided only for testing the workflow.
* UMA, MACE, SevenNet, and ORB-v3 can download their default model weights automatically when supported by the package.
* For NequIP, compiled model file is required. example file is given
* EquiformerV3 requires the external EquiformerV3 source code and configuration.
