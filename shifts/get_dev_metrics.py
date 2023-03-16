import glob
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from prettytable import PrettyTable
import pandas as pd

legend ={
    'unet-3d-baseline': "UNet3D",
    'unet-monai-baseline': "UnetMonAI",
    'unet-attn-rsa-baseline': "UnetRSA",
    'unet-rsa-baseline_dec_first_pos_1234': "RSA-D-4",
    'unet-rsa-baseline_enc_first_sec_pos_1234': "RSA-E-FS",
    'unet-rsa-baseline_enc_dec_first_pos_1234': "RSA-ED-4",
    'unet-rsa-baseline_enc_first_pos_1234': "RSA-4",
    'unet-rsa-baseline_enc_first_pos_1234_enc_first_pos_24': "RSA-2",
    'unet-tr-baseline': "UNETR",
    'unet-attention-baseline': "AttnUnet",
    'unet-dy-baseline': "DyNet",
    'unet-res-baseline': "ResUnet",
    'segresnet-monai-baseline': "SegRes",
    'vnet-baseline': "VNet",
}

files = sorted(glob.glob("logs/**/**/metrics.jsonl"))

# dd = defaultdict(list)
# fname = files[0]
# for fname in files:
#     parts = Path(fname).parts
#     run_name = parts[-3]
#     with open(fname, "r") as f:
#         json_list = list(f)
#         result = [json.loads(json_str)["metric"] for json_str in json_list if "metric" in json_str]
#         result = np.array(result)
#     dd[run_name].append(result.max())

# scores = {k: f"{np.mean(v):.3f}" for k, v in dd.items()}
# pprint(scores)
# scores_rows = [[legend[k], v] for k, v in scores.items()]
# x = PrettyTable()
# header = ["Model", "DICE on dev_in"]
# x.field_names = header
# x.add_rows(scores_rows)
# print(x)

# df = pd.DataFrame(scores_rows, columns=header)
# print(df.sort_values(header[-1]))

# met_dd = defaultdict(lambda: defaultdict(list))
# files = sorted(glob.glob("logs/**/**/eval_in_scores.npz"))
# fname = files[0]
# headers = []
# for fname in files:
#     parts = Path(fname).parts
#     run_name = parts[-3]
#     x = np.load(fname)
#     metrics = {k: x[k].mean() for k in list(x.keys())}
#     headers = list(metrics.keys())
#     [met_dd[run_name][k].append(metrics[k]) for k in metrics]

# eval_in_scores = {k: {v1: np.round(np.mean(v2), 3) for v1, v2 in v.items()} for k, v in met_dd.items()}
# scores_rows = [[legend[k]] + list(v.values()) for k, v in eval_in_scores.items()]
# x = PrettyTable()
# x.field_names = ["Model"] + headers
# x.add_rows(scores_rows)
# print(x)

def calculate_metrics(metric_name):
    met_dd = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(f"logs/**/**/{metric_name}_scores.npz"))
    headers = []
    for fname in files:
        parts = Path(fname).parts
        run_name = parts[-3]
        x = np.load(fname)
        metrics = {k: x[k].mean() for k in list(x.keys())}
        headers = list(metrics.keys())
        [met_dd[run_name][k].append(metrics[k]) for k in metrics]

    # metric_scores = {k: {v1: np.round(np.mean(v2), 3) for v1, v2 in v.items()} for k, v in met_dd.items()}
    metric_scores = {k: {v1: np.round(np.max(v2), 3) for v1, v2 in v.items()} for k, v in met_dd.items()}
    scores_rows = [[legend[k]] + list(v.values()) for k, v in metric_scores.items()]
    x = PrettyTable()
    x.field_names = ["Model"] + headers
    x.add_rows(scores_rows)
    print(f"Evaluating on {metric_name}")
    print(x)
    print("\n")
    return [["Model"] + headers] + scores_rows, metric_scores

names = ["dev_in", "dev_out", "eval_in"]

dev_in, dev_in_metrics = calculate_metrics("dev_in")
dev_out, dev_out_metrics = calculate_metrics("dev_out")
eval_in, eval_in_metrics = calculate_metrics("eval_in")
dev_in = np.array(dev_in)
dev_out = np.array(dev_out)
eval_in = np.array(eval_in)
scores = np.concatenate([dev_in, dev_out, eval_in], axis=-1)

chosen_cols = [0, 1, 5, 9, 3, 7, 11]
chosen_scores = scores[..., chosen_cols]
field_names = [chosen_scores[0, 0]] + [ f"{n}_{csh}" for n, csh in zip(names + names, chosen_scores[0, 1:])]
x = PrettyTable()
x.field_names = field_names
x.add_rows(chosen_scores[1:])
print(x)

chosen_cols = [0, 1, 5, 9]
chosen_scores = scores[..., chosen_cols]
field_names = [chosen_scores[0, 0]] + [ f"{n}_{csh}" for n, csh in zip(names, chosen_scores[0, 1:])]
x = PrettyTable()
x.field_names = field_names
x.add_rows(chosen_scores[1:])
print(x)

df = pd.DataFrame(chosen_scores[1:], columns=field_names)
print(df.sort_values(field_names[1])["Model"])
print(df.sort_values(field_names[2])["Model"])
print(df.sort_values(field_names[3])["Model"])


