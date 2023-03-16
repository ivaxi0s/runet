from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
import torch
import logging
import os
import re
from tensorboardX import SummaryWriter
from attn_rsa_unet import AttentionRSAUnet

from unet3d import UNet3D
from monai.networks.nets import SegResNet, UNet, UNETR, AttentionUnet, DynUNet, BasicUNetPlusPlus, VNet

def create_model(args, roi_size=(96, 96, 96)):
    if args.model == "unet-monai":
        model = UNet(spatial_dims=3,
                    in_channels=1,
                    out_channels=2,
                    channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),
                    num_res_units=0)

    elif args.model == "unet-tr":
            model = UNETR(
                    in_channels=1,
                    out_channels=2,
                    img_size=roi_size,
                    feature_size=16,
                    num_heads=8,
                    norm_name='batch',
                    spatial_dims=3)
    elif args.model =="unet-res":
        model = UNet(spatial_dims=3,
                    in_channels=1,
                    out_channels=2,
                    num_res_units=2,
                    channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),
        )
    elif args.model =="unet-attention":
        model = AttentionUnet(spatial_dims=3,
                    in_channels=1,
                    out_channels=2,
                    channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),
                    )
    elif args.model == "vnet":
        model = VNet(spatial_dims=3,
                    in_channels=1,
                    out_channels=2
                    )
    elif args.model == "unet-attn-rsa":
        model = AttentionRSAUnet(spatial_dims=3,
                    in_channels=1,
                    out_channels=2,
                    channels=(32, 64, 128, 256, 512),
                    strides=(2, 2, 2, 2),
                    )
    elif args.model =="unet-dy":
        model = DynUNet(spatial_dims=3,
                    in_channels=1,
                    out_channels=2,
                    kernel_size=(3, 3, 3, 3),
                    strides=(1, 2, 2, 2),
                    upsample_kernel_size=(2, 2, 2, 2)
                    )
    elif args.model == "unet-3d":
        model = UNet3D(n_in=1, n_out=2)
    elif args.model == "unet-rsa":
        model = UNet3D(
                    n_in=1,
                    n_out=2,
                    use_rsa_enc=args.rsa_enc,
                    use_rsa_dec=args.rsa_dec,
                    use_rsa_first=args.rsa_first,
                    use_rsa_second=args.rsa_second,
                    use_rsa_pos=args.rsa_pos)
    elif args.model == "segresnet-monai":
            model = SegResNet(
                        spatial_dims=3,
                        in_channels=1,
                        out_channels=2,
                        init_filters=16,
                        blocks_down=[1, 2, 2, 4],
                        blocks_up=[1, 1, 1],
                        dropout_prob=0.2,
                    )
    else:
        raise ValueError(f"Invalid Model: {args.model}")

    return model


def load_model(model, args, device):
    logging.info(f'Model loaded from {args.checkpoint}')
    loaded_checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in loaded_checkpoint:
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        last_epoch = loaded_checkpoint['epoch'] if 'epoch' in loaded_checkpoint else 0
    else:
        model.load_state_dict(loaded_checkpoint)
        filename = os.path.basename(args.checkpoint)
        try:
            last_epoch = int(re.compile(r'\d\d').findall(filename)[-1])
        except (IndexError, ValueError, TypeError) as e:
            logging.info(f"Resume is not possible --> {e}")
            last_epoch = 0

    last_epoch = last_epoch if args.resume else 0
    logging.info(f"Resuming from epoch {last_epoch}")
    args.batch_size = loaded_checkpoint['batch_size'] if 'batch_size' in loaded_checkpoint else args.batch_size
    args.learning_rate = loaded_checkpoint['lr'] if 'lr' in loaded_checkpoint else args.learning_rate
    args.lr_decay = loaded_checkpoint['lr_decay'] if 'lr_decay' in loaded_checkpoint else args.lr_decay

    return last_epoch

def num_params(model):
    return sum(p.numel() for p in model.parameters())

class Logger:
    def __init__(self, logdir: Path, step: int):
        self._logdir = Path(logdir)
        self.writer = SummaryWriter(log_dir=str(self._logdir))
        self._scalars = defaultdict(list)
        self._images = {}
        self._videos = {}
        self._last_step = None
        self._last_time = None
        self.step = step

    def scalar(self, name, value):
        value = float(value)
        self._scalars[name].append(value)

    def write(self, fps=False):
        scalars = {k: np.mean(v) for k, v in self._scalars.items()}
        scalars = list(scalars.items())
        if len(scalars) == 0:
            return
        if fps:
            scalars.append(("perf/fps", self._compute_fps(self.step)))
        video_fps = 5
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": self.step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            prefix = "" if "/" in name else "scalars/"
            self.writer.add_scalar(prefix + name, np.mean(value), self.step)

        self._scalars = defaultdict(list)
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def close(self):
        self.writer.close()

