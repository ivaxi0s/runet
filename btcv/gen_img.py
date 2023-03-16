# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np
import torch, pdb
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader
import matplotlib.pyplot as plt
from segmentation_mask_overlay import overlay_masks
import seaborn as sns

import torchio as tio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from monai.inferers import sliding_window_inference

from networks.model_utils import Logger, create_model, num_params

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--saveimg", action="store_true", help="convert images to niigz")
parser.add_argument("--imgfolder", default="files", type=str, help="folder to save the nii.gz images")


parser.add_argument('--model', type=str, default="unet-tr", help='Specify the global random seed')


# RSA params
parser.add_argument('--rsa', action='store_true', help='Enables RSA')
parser.add_argument('--rsa-enc', action='store_true', help='Enables RSA at the Encoder')
parser.add_argument('--rsa-dec', action='store_true', help='Enables RSA at the Decoder')
parser.add_argument('--rsa-first', action='store_true', help='Enables RSA at the first layer of DoubleConv Block')
parser.add_argument('--rsa-second', action='store_true', help='Enables RSA at the second layer of DoubleConv Block')
parser.add_argument('--rsa-pos', nargs='+', default=[1, 2, 3, 4], type=int, help='Define RSA positions')

parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')

def main():
    args = parser.parse_args()
    args.test_mode = True
    args.leaderboard = False
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = create_model(args)

        model = torch.load(pretrained_pth)
        # model.load_state_dict(model_dict)

    model.eval()
    model.to(device)
    additional_model_paths = [
        os.path.join(pretrained_dir, 'unet-monai_1_model.pth'),
        os.path.join(pretrained_dir, 'unet-tr_1_model.pth'),
        os.path.join(pretrained_dir, 'unet-res_1_model.pth')

    ]
    slice_map = {
    "img0035.nii.gz": 170,
    "img0036.nii.gz": 230,
    "img0037.nii.gz": 204,
    "img0038.nii.gz": 204,
    "img0039.nii.gz": 185,
    "img0040.nii.gz": 180,
    }
    case_num = 4
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # plt.figure("check", (18, 6))
            if i == 4:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)


                plt.figure("check", (18, 6))
                plt.rcParams['text.usetex'] = True

                plt.subplot(1, 6, 1)
                plt.title("CT Image", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))

                plt.xticks([])
                plt.yticks([])

                plt.subplot(1, 6, 2)
                plt.title("Label", fontsize=25)
                cmap = ListedColormap([[0, 0, 0, 0.0]] + sns.color_palette(n_colors=13))
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                
                
                plt.subplot(1, 6, 3)
                plt.title("RUNet", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("85.55", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                dice_list_sub = []
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))

                model = torch.load(additional_model_paths[0])
                model.eval()
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
                plt.subplot(1, 6, 4)
                plt.title("UNet", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("79.59", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # dice_list_sub = []
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))
                
                model = torch.load(additional_model_paths[1])
                model.eval()
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
                plt.subplot(1, 6, 5)
                plt.title("UNETR", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("78.13", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                # dice_list_sub = []
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))

                model = torch.load(additional_model_paths[1])
                model.eval()
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                plt.subplot(1, 6, 6)
                plt.title("nnUNet", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("83.81", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # dice_list_sub = []
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))
                # plt.show()
                plt.tight_layout()
                file_location = os.path.join(args.imgfolder, 'fin4.pdf')
                plt.savefig(file_location, bbox_inches="tight", dpi=200)
                file_location = os.path.join(args.imgfolder, 'fin4')
                plt.savefig(file_location, bbox_inches="tight", dpi=200)
                break
            if i == 5:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)

                slice_map[img_name] = 170
                plt.figure("check", (18, 6))
                # plt.rcParams['text.usetex'] = True

                plt.subplot(1, 6, 1)
                plt.title("CT Image", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))

                plt.xticks([])
                plt.yticks([])

                plt.subplot(1, 6, 2)
                plt.title("Label", fontsize=25)
                cmap = ListedColormap([[0, 0, 0, 0.0]] + sns.color_palette(n_colors=13))
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                
                
                plt.subplot(1, 6, 3)
                plt.title("RUNet", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("85.55", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                dice_list_sub = []
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))

                model = torch.load(additional_model_paths[0])
                model.eval()
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
                plt.subplot(1, 6, 4)
                plt.title("UNet", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("79.59", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # dice_list_sub = []
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))
                
                model = torch.load(additional_model_paths[1])
                model.eval()
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
                plt.subplot(1, 6, 5)
                plt.title("UNETR", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("78.13", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                # dice_list_sub = []
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))

                model = torch.load(additional_model_paths[1])
                model.eval()
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                plt.subplot(1, 6, 6)
                plt.title("nnUNet", fontsize=25)
                plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap=cmap, alpha=0.8)
                # plt.xlabel("83.81", fontsize=25)
                # plt.gca().add_patch(Rectangle((95,20),15,15,
                #     edgecolor='red',
                #     facecolor='none',
                #     lw=2.5))
                plt.xticks([])
                plt.yticks([])
                # val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                # val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                # dice_list_sub = []
                # for i in range(1, 14):
                #     organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                #     dice_list_sub.append(organ_Dice)
                # mean_dice = np.mean(dice_list_sub)
                # print("Mean Organ Dice: {}".format(mean_dice))
                # plt.show()
                plt.tight_layout()
                file_location = os.path.join(args.imgfolder, 'fin3.pdf')
                plt.savefig(file_location, bbox_inches="tight", dpi=200)
                file_location = os.path.join(args.imgfolder, 'fin3')
                plt.savefig(file_location, bbox_inches="tight", dpi=200)
                break


if __name__ == "__main__":
    main()
