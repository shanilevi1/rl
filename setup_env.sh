#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --gres=gpu:1
#SBATCH --partition=p_b200_schwartz
#SBATCH --mem=32G
#SBATCH --account=ug_schwartz
#SBATCH --time=01:00:00
#SBATCH --output=setup_env.log
#SBATCH --error=setup_env.err

source ~/.bashrc
set -e

ENV_PATH="/private/schwartz-lab/weidena1/envs/rl-1"

echo "==> Removing existing env (if any)..."
conda env remove -p "$ENV_PATH" -y 2>/dev/null || true

echo "==> Creating new env at $ENV_PATH..."
conda create -p "$ENV_PATH" python=3.10 -y

echo "==> Installing PyTorch 2.7.0+cu128 (first stable release with B200/Blackwell support)..."
conda run -p "$ENV_PATH" pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing remaining requirements via pip..."
conda run -p "$ENV_PATH" pip install \
    "gymnasium[atari,accept-rom-license]" \
    ale-py \
    opencv-python-headless \
    wandb \
    pyyaml \
    tqdm

echo "==> Submitting SLURM jobs..."
cd /private/schwartz-lab/weidena1/rl
sbatch run_0.sh
sbatch run_1.sh
sbatch run_2.sh

echo "==> Done. Jobs submitted."
