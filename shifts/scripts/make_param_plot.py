import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams['text.usetex'] = True

df = pd.read_csv("scripts/acc.csv")
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

model_names = ["unet-monai", "unet-tr", "unet-res", "unet-attn-rsa", "unet-dy", "unet-3d"]

new_df = pd.read_csv("scripts/param_data.csv")
params = dict(zip(new_df.Model, new_df.Params))

df["Model"] = df["model_name"].map(legend)
df['params'] = df['model_name'].map(params)
df = df.dropna(axis=0)
axes = ["dev_in_ndsc", "dev_out_ndsc", "eval_in_ndsc"]

titles={
    "dev_in_ndsc": r"$Dev_{in}$", 
    "dev_out_ndsc": r"$Dev_{out}$", 
    "eval_in_ndsc": r"$Eval_{in}$"
}
fig = plt.figure(figsize=(8, 3))
for col, axis in enumerate(axes):
    x = df["params"].tolist()
    y = df[axis].tolist()
    labels = df["Model"].tolist()
    colors = sns.color_palette(n_colors=len(params))
    plt.subplot(1, 3, col + 1)
    if col == 0:
        plt.ylabel("nDSC")
    plt.xlabel("Num Parameters")
    plt.title(titles[axis])
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=colors[i], label=labels[i])
        plt.annotate(labels[i], (x[i] + (0.05e8), y[i]-0.1), fontsize= 'medium')
    plt.xlim(0, 1.2e8)

fname="plots/params.png"
plt.savefig(fname, bbox_inches="tight", dpi=200)
print(f"Saving plot to {fname}")
fname="plots/params.pdf"
plt.savefig(fname, bbox_inches="tight", dpi=200)
print(f"Saving plot to {fname}")
plt.close()

