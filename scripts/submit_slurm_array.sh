#!/bin/bash
#SBATCH --job-name=mlip_relax
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

CONFIG=${1:-configs/mace.yaml}
mkdir -p logs

source ~/.bashrc
conda activate mlip-screening

python scripts/run_relaxations.py --config "$CONFIG"
