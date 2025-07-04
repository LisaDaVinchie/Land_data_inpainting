#!/bin/bash

# Configurable wait time
WAIT_BUDGET_MINUTES=1

# Submit GPU job
gpu_jobid=$(sbatch job_gpu.sh | awk '{print $4}')
echo "Submitted GPU job: ${gpu_jobid}"

# Watch for wait budget
(
    sleep ${WAIT_BUDGET_MINUTES}m
    state=$(squeue -j ${gpu_jobid} -h -o "%T")
    if [ "$state" == "PENDING" ]; then
        echo "GPU job still pending after ${WAIT_BUDGET_MINUTES} minutes. Cancelling and submitting CPU job."
        scancel ${gpu_jobid}
        cpu_jobid=$(sbatch job_cpu.sh | awk '{print $4}')
        echo "Submitted CPU fallback job: ${cpu_jobid}"
    else
        echo "GPU job has started or completed within wait budget."
    fi
) &
