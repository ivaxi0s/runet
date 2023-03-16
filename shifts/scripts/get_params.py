import time
from model_utils import create_model, num_params
from addict import Dict
import pandas as pd
from ptflops import get_model_complexity_info
import torch

inp_size = (2, 1, 96, 96, 96)
inp = torch.randn(inp_size)

legend ={
    'unet-3d': "UNet3D",
    'unet-monai': "UnetMonAI",
    'unet-attn-rsa': "RUnet",
    'unet-rsa': "RSA-4",
    'unet-tr': "UNETR",
    'unet-attention': "AttnUnet",
    'unet-dy': "DyNet",
    'unet-res': "ResUnet",
    'segresnet-monai': "SegRes",
    "vnet": "VNet"
}

args = Dict()
args.rsa_enc=True
args.rsa_dec=False
args.rsa_first=True
args.rsa_second=False
args.rsa_pos=[1, 2, 3, 4]

model_names = ["unet-monai", "unet-tr", "unet-res", "unet-attention", "vnet", "unet-attn-rsa", "unet-dy", "unet-3d", "unet-rsa"]

times = {}
params = {}
flops = {}
for model_name in model_names:
    print(f"{model_name}")
    args.model = model_name
    model = create_model(args)
    model.eval();
    _ = model(inp)
    start = time.time()
    _ = model(inp)
    times[model_name] = time.time() - start
    macs, _ = get_model_complexity_info(model, inp_size[1:], as_strings=False,
                                           print_per_layer_stat=False)
    flops[model_name] = macs
    params[model_name] = num_params(model)

new_df = pd.DataFrame({"Model": params.keys(), "Params": params.values(), "Flops": flops.values(), "Times": times.values()})
new_df.to_csv("scripts/param_data.csv")


