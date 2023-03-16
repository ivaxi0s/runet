#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --array=1-3
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

seed=${SLURM_ARRAY_TASK_ID}

data_dir="~/scratch/rsa_data/shifts_ms_pt1/shifts_ms_pt1/msseg"
data_args="--path_train_data ${data_dir}/train/flair --path_train_gts ${data_dir}/train/gt --path_val_data ${data_dir}/dev_in/flair --path_val_gts ${data_dir}/dev_in/gt"

ID="baseline"

# model="unet-monai"

model="unet-rsa"
rsa_args="--rsa --rsa-enc --rsa-dec --rsa-first --rsa-second --rsa-pos 1 2 3 4"
rsa_args="--rsa --rsa-enc --rsa-first --rsa-pos 1 2 3 4"
ID="${ID}_enc_first_pos_1234"
test_args="${test_args} ${rsa_args}"

# model="unet-tr"

# model="unet-3d"

python train.py --model ${model} --id ${ID} --seed $seed --path_save ${logdir} ${data_args} ${test_args}