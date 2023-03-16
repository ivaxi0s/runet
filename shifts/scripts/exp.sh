#!/bin/bash
#SBATCH --time=17:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --array=1-2
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

# 4hrs
# model="unet-monai"

# 18hrs
# model="unet-rsa"
# 28hrs
# rsa_args="--rsa --rsa-dec --rsa-first --rsa-pos 1 2 3 4"
# ID="${ID}_dec_first_pos_1234"
# 45hrs
# rsa_args="--rsa --rsa-dec --rsa-first --rsa-second --rsa-pos 1 2 3 4"
# ID="${ID}_dec_first_sec_pos_1234"
# 30hrs
# rsa_args="--rsa --rsa-enc --rsa-first --rsa-second --rsa-pos 1 2 3 4"
# ID="${ID}_enc_first_sec_pos_1234"
# rsa_args="--rsa --rsa-enc --rsa-dec --rsa-first --rsa-pos 1 2 3 4"
# ID="${ID}_enc_dec_first_pos_1234"
# rsa_args="--rsa --rsa-enc --rsa-first --rsa-pos 1 2 3 4"
# ID="${ID}_enc_first_pos_1234"
# rsa_args="--rsa --rsa-enc --rsa-first --rsa-pos 2 4"
# ID="${ID}_enc_first_pos_24"
# test_args="${test_args} ${rsa_args}"

# 18hrs
model="vnet"

# 36hrs
# model="unet-attn-rsa"

# 14hrs
# model="unet-tr"

# 14hrs
# model="unet-3d"

# 8hrs
# model="unet-res"

# 15hrs
# model="unet-attention"

# 42hrs
# model="unet-plusplus"

# 18hrs
# model="unet-dy"

# 12hrs
# model="segresnet-monai"

python train.py --model ${model} --id ${ID} --seed $seed --path_save ${logdir} ${data_args} ${test_args}