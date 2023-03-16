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
SLURM_ARRAY_TASK_ID="3"

exp_no=${SLURM_ARRAY_TASK_ID}

test_args="--num_workers 4"

ID="baseline"

# model="unet-monai"

# model="unet-rsa"
# * Options
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
# ! Should be uncommented for RSA
# test_args="${test_args} ${rsa_args}"

# 43hrs
# model="unet-attn-rsa"

# model="vnet"

# model="unet-tr"

# model="unet-3d"

# 8hrs
# model="unet-res"

# 15hrs
# model="unet-attention"

# 42hrs
# model="unet-plusplus"

# 18hrs
model="unet-dy"

# 12hrs
# model="segresnet-monai"

# data_dir="~/scratch/rsa_data/shifts_ms_pt1/shifts_ms_pt1/msseg/eval_in"
# data_args="--path_data_name eval_in --path_data ${data_dir}/flair --path_gts ${data_dir}/gt --path_bm ${data_dir}/fg_mask"
# python test.py --model ${model} --id ${ID} --path_model ${logdir} --threshold 0.35 ${data_args} ${test_args}

# data_dir="~/scratch/rsa_data/shifts_ms_pt1/shifts_ms_pt1/msseg/dev_in"
# data_args="--path_data_name dev_in --path_data ${data_dir}/flair --path_gts ${data_dir}/gt --path_bm ${data_dir}/fg_mask"
# python test.py --model ${model} --id ${ID} --path_model ${logdir} --threshold 0.35 ${data_args} ${test_args}

# data_dir="~/scratch/rsa_data/shift_ms/shifts_ms_pt2/ljubljana/dev_out"
# data_args="--path_data_name dev_out --path_data ${data_dir}/flair --path_gts ${data_dir}/gt --path_bm ${data_dir}/fg_mask"
# python test.py --model ${model} --id ${ID} --path_model ${logdir} --threshold 0.35 ${data_args} ${test_args}

if [ $exp_no -eq 1 ]
then
data_dir="~/scratch/rsa_data/shifts_ms_pt1/shifts_ms_pt1/msseg/eval_in"
data_args="--path_data_name eval_in --path_data ${data_dir}/flair --path_gts ${data_dir}/gt --path_bm ${data_dir}/fg_mask"

elif [ $exp_no -eq 2 ]
then
data_dir="~/scratch/rsa_data/shifts_ms_pt1/shifts_ms_pt1/msseg/dev_in"
data_args="--path_data_name dev_in --path_data ${data_dir}/flair --path_gts ${data_dir}/gt --path_bm ${data_dir}/fg_mask"

elif [ $exp_no -eq 3 ]
then
data_dir="~/scratch/rsa_data/shift_ms/shifts_ms_pt2/ljubljana/dev_out"
data_args="--path_data_name dev_out --path_data ${data_dir}/flair --path_gts ${data_dir}/gt --path_bm ${data_dir}/fg_mask"

else
echo -e "Not a valid experiment number"
exit

fi

echo -e "${data_args}"
python get_segmentation_masks.py --model ${model} --id ${ID} --path_model ${logdir} --threshold 0.35 ${data_args} ${test_args}
