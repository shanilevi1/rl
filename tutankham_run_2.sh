#!/bin/bash
#SBATCH --job-name=tut_run_2
#SBATCH --gres=gpu:1
#SBATCH --partition=p_rtx_schwartz
#SBATCH --mem=200G
#SBATCH --account=ug_schwartz
#SBATCH --time=48:00:00
#SBATCH --output=runs/tutankham_2/slurm_%j.log
#SBATCH --error=runs/tutankham_2/slurm_%j.err

source ~/.bashrc
conda activate /private/schwartz-lab/weidena1/envs/rl-1
export LD_LIBRARY_PATH=/private/schwartz-lab/weidena1/envs/rl-1/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

cd /private/schwartz-lab/weidena1/rl
[ -f runs/tutankham_2/training_stats.jsonl ] && mv runs/tutankham_2/training_stats.jsonl runs/tutankham_2/training_stats_prev.jsonl
python train.py --run_id tutankham_2 --seed 2 --config configs/tutankham_base.yaml
