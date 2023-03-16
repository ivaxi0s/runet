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
import torch
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader

import torchio as tio

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
parser.add_argument("--nifolder", default="files", type=str, help="folder to save the nii.gz images")


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
    args.test_mode = False
    args.leaderboard = True
    
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
        # import pdb; pdb.set_trace()
        # model = create_model(args)

        model_dict = torch.load(pretrained_pth)
        model.load_state_dict(model_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs = (batch["image"].cuda())
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            if args.saveimg:
                file_location = os.path.join(args.nifolder, args.model)
                if not os.path.exists(file_location): os.makedirs(file_location)
                filename = os.path.join(file_location, img_name)
                img_nii = tio.LabelMap(tensor=val_outputs)
                img_nii.save(filename)

if __name__ == "__main__":
    main()
