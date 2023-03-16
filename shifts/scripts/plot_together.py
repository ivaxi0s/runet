from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['text.usetex'] = True

legend ={
    'unet-3d': "UNet*",
    'unet-monai': "UNet",
    'unet-attn-rsa': "RUNet",
    'unet-rsa': "RSA-4",
    'unet-tr': "UNETR",
    'unet-attention': "AttnUNet",
    'unet-dy': "nnUNet",
    'unet-res': "ResUNet",
    'segresnet-monai': "SegRes",
    "vnet": "VNet"
}

model_names = ["unet-monai", "unet-tr", "unet-res", "unet-attn-rsa", "unet-dy", "unet-3d"]

new_df = pd.read_csv("scripts/param_data.csv")
params = dict(zip(new_df.Model, new_df.Params))

df = pd.read_csv("scripts/acc.csv")
df_dict = dict(zip(df.model_name, df.dev_out_ndsc))
x_data = OrderedDict({legend[k]: params[k] for k in model_names})
y_data = OrderedDict({legend[k]: df_dict[k] for k in model_names})
x_offset = 0.05e8
y_offset = -0.1
fontsize= 'medium'

fig = plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
x = list(x_data.values())
y = list(y_data.values())
labels = [legend[k] for k in model_names]
colors = sns.color_palette(n_colors=len(model_names))
plt.ylabel("nDSC (\%)")
plt.xlabel("Parameters")
plt.title(r"SHIFTS $Dev_{out}$")
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], label=labels[i])
    plt.annotate(labels[i], (x[i] + x_offset, y[i] + y_offset), fontsize=fontsize)
plt.xlim(0, 1.3e8)

model_names = ["unet-monai", "unet-tr", "unet-res", "unet-attn-rsa", "unet-dy", "unet-3d", "unet-attention"]
df = pd.read_csv("scripts/btcv_acc.csv")
df_dict = dict(zip(df.model_name, df.Dice))
x_data = OrderedDict({legend[k]: params[k] for k in model_names})
y_data = OrderedDict({legend[k]: df_dict[k] for k in model_names})
plt.subplot(1, 2, 2)
x = list(x_data.values())
y = list(y_data.values())
labels = [legend[k] for k in model_names]
colors = sns.color_palette(n_colors=len(model_names))
plt.ylabel("DICE (\%)")
plt.xlabel("Parameters")
plt.title(r"BTCV $DICE$")
for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], label=labels[i])
    plt.annotate(labels[i], (x[i] + x_offset, y[i] + y_offset), fontsize=fontsize)
plt.xlim(0, 1.3e8)

fname="plots/params.png"
plt.savefig(fname, bbox_inches="tight", dpi=200)
print(f"Saving plot to {fname}")
fname="plots/params.pdf"
plt.savefig(fname, bbox_inches="tight", dpi=200)
print(f"Saving plot to {fname}")
plt.close()

