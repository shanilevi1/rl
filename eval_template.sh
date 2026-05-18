#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=p_b200_schwartz
#SBATCH --mem=32G
#SBATCH --account=ug_schwartz
#SBATCH --time=2:00:00

RUN_ID=$1
CONFIG=$2
EPISODES=${3:-100}

source ~/.bashrc
conda activate /private/schwartz-lab/weidena1/envs/rl-1
export LD_LIBRARY_PATH=/private/schwartz-lab/weidena1/envs/rl-1/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

cd /private/schwartz-lab/weidena1/rl
python eval_run.py --run_id ${RUN_ID} --config ${CONFIG} --episodes ${EPISODES}
