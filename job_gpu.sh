#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source bot_codes.txt
bot_id=${bot_id}
chat_id=${chat_id}

# source wandb_api.txt
# wandb_api=${api_code}

# export WANDB_API_KEY=${wandb_api}

set -e

notify_telegram() {
    local status="$1"
    curl -s -X POST "https://api.telegram.org/bot${bot_id}/sendMessage" \
    -d chat_id=${chat_id} \
    -d text="Your Land_data_inpainting slurm job (Job ID: $SLURM_JOB_ID) has completed with status: $status"
}

trap 'notify_telegram "FAILED (job terminated or timed out)"' TERM EXIT

if [ -z "$VIRTUAL_ENV" ]; then
    source venv_inpainting/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
fi

ulimit -a

PYTHON=$(which python3)

$PYTHON src/train.py || { echo "Training failed"; exit 1; }

notify_telegram "SUCCESS"

# Clear the EXIT trap to avoid double notifications
trap - EXIT

echo "Training completed successfully"
