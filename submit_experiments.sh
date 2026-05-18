#!/bin/bash
# Submit 7 experiments × 2 seeds = 14 jobs, max 6 GPUs at a time.
# Jobs are submitted in batches of 6. Each batch waits for the previous to finish.
# Usage: bash submit_experiments.sh

set -e
cd /private/schwartz-lab/weidena1/rl

submit_job() {
    local run_id=$1
    local seed=$2
    local config=$3
    local dep=$4

    mkdir -p runs/${run_id}

    local dep_flag=""
    [ -n "$dep" ] && dep_flag="--dependency=afterany:${dep}"

    sbatch \
        --job-name=${run_id} \
        --output=runs/${run_id}/slurm_%j.log \
        --error=runs/${run_id}/slurm_%j.err \
        ${dep_flag} \
        run_template.sh ${run_id} ${seed} ${config} \
        | awk '{print $4}'
}

# ── Batch 1: 6 jobs, run immediately ─────────────────────────────────────────
echo "==> Batch 1 (runs immediately):"
j1=$(submit_job exp_03_s0 0 configs/ablations/exp_03_eps_end05.yaml)
j2=$(submit_job exp_03_s2 2 configs/ablations/exp_03_eps_end05.yaml)
j3=$(submit_job exp_04_s0 0 configs/ablations/exp_04_big_buffer.yaml)
j4=$(submit_job exp_04_s2 2 configs/ablations/exp_04_big_buffer.yaml)
j5=$(submit_job exp_05_s0 0 configs/ablations/exp_05_batch64.yaml)
j6=$(submit_job exp_05_s2 2 configs/ablations/exp_05_batch64.yaml)
echo "  exp_03 s0=$j1 s2=$j2 | exp_04 s0=$j3 s2=$j4 | exp_05 s0=$j5 s2=$j6"

BATCH1="${j1}:${j2}:${j3}:${j4}:${j5}:${j6}"

# ── Batch 2: 6 jobs, wait for ALL of batch 1 ────────────────────────────────
echo "==> Batch 2 (waits for batch 1):"
j7=$(submit_job  exp_06_s0 0 configs/ablations/exp_06_lr_0002.yaml      "$BATCH1")
j8=$(submit_job  exp_06_s2 2 configs/ablations/exp_06_lr_0002.yaml      "$BATCH1")
j9=$(submit_job  exp_07_s0 0 configs/ablations/exp_07_fast_anneal.yaml  "$BATCH1")
j10=$(submit_job exp_07_s2 2 configs/ablations/exp_07_fast_anneal.yaml  "$BATCH1")
j11=$(submit_job exp_08_s0 0 configs/ablations/exp_08_clip5_eps05.yaml  "$BATCH1")
j12=$(submit_job exp_08_s2 2 configs/ablations/exp_08_clip5_eps05.yaml  "$BATCH1")
echo "  exp_06 s0=$j7 s2=$j8 | exp_07 s0=$j9 s2=$j10 | exp_08 s0=$j11 s2=$j12"

BATCH2="${j7}:${j8}:${j9}:${j10}:${j11}:${j12}"

# ── Batch 3: 2 jobs, wait for ALL of batch 2 ────────────────────────────────
echo "==> Batch 3 (waits for batch 2):"
j13=$(submit_job exp_09_s0 0 configs/ablations/exp_09_stability_combo.yaml "$BATCH2")
j14=$(submit_job exp_09_s2 2 configs/ablations/exp_09_stability_combo.yaml "$BATCH2")
echo "  exp_09 s0=$j13 s2=$j14"

echo ""
echo "All 14 jobs queued in 3 batches. Max 6 GPUs used at once."
echo "Check queue: squeue -u \$USER"
