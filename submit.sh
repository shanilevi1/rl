#!/bin/bash
# Submit all 3 DQN training runs in parallel.
# Usage: bash submit.sh

sbatch run_0.sh
sbatch run_1.sh
sbatch run_2.sh
