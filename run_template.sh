#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=p_b200_schwartz
#SBATCH --mem=200G
#SBATCH --account=ug_schwartz
#SBATCH --time=48:00:00

RUN_ID=$1
SEED=$2
CONFIG=$3

source ~/.bashrc
conda activate /private/schwartz-lab/weidena1/envs/rl-1
export LD_LIBRARY_PATH=/private/schwartz-lab/weidena1/envs/rl-1/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

cd /private/schwartz-lab/weidena1/rl
[ -f runs/${RUN_ID}/training_stats.jsonl ] && mv runs/${RUN_ID}/training_stats.jsonl runs/${RUN_ID}/training_stats_prev.jsonl
python train.py --run_id ${RUN_ID} --seed ${SEED} --config ${CONFIG}
