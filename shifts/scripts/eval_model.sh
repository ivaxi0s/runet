#!/bin/bash
#SBATCH --time=35:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --array=1-1
#SBATCH --job-name=rsa
#SBATCH --output=out/%x_%A_%a.out
#SBATCH --error=out/%x_%A_%a.err
#SBATCH --mail-user=shivakanth.sujit@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

source ~/rsa_env/bin/activate

logdir="logs"

# logdir="logs/testing"
# SLURM_ARRAY_TASK_ID="1"

# seed=${SLURM_ARRAY_TASK_ID}

test_args="--num_workers 4 --n_jobs 2"

data_dir="~/scratch/rsa_data/shifts_ms_pt1/shifts_ms_pt1/msseg"
data_args="--path_data ${data_dir}/eval_in/flair --path_gts ${data_dir}/eval_in/gt --path_bm ${data_dir}/eval_in/fg_mask"

model="unet-monai"
ID="baseline"

python test.py --model ${model} --id ${ID} --path_model ${logdir} --threshold 0.35 ${data_args} ${test_args}