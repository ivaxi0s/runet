import glob
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
from metrics import dice_norm_metric
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

legend ={f"{k}-baseline": v for k, v in legend.items()}

split = "dev_out"
files = glob.glob(f"logs/**/seed1/images/{split}_pred.npz")

images = {}

for fname in files:
    x = np.load(fname)
    parts = Path(fname).parts
    run_name = parts[-4]
    images[run_name] = {k: x[k] for k in list(x.keys())}

chosen_method = "unet-attn-rsa-baseline"
baselines = [x for x in images.keys() if chosen_method not in x]

ndsc_mets = {}
metrics = {}
for run_name, image_dict in images.items():
    gt = image_dict["gt"]
    seg = image_dict["pred"]
    ndsc_mets[run_name] = dice_norm_metric(ground_truth=gt, predictions=seg)
    metrics[run_name] = np.array([dice_norm_metric(ground_truth=gt[..., i], predictions=seg[..., i]) for i in range(seg.shape[-1])])

ct_img = images[chosen_method]["image"]
input_img = images[chosen_method]["gt"]

# segss = np.array([input_img[..., i].sum() for i in range(input_img.shape[-1])])
# chosen_slices = np.logical_and(metrics[chosen_method] != 1, segss != 0)
# chosen_images = {k: v[chosen_slices] for k, v in metrics.items()}
# d = chosen_images[chosen_method]
# # chosen_data = np.array([v[chosen_slices] for k, v in metrics.items()])

# diffs = np.array([d - chosen_images[b] for b in baselines])

# good_indices = (diffs > 0).mean(0) == 1
# good_indices = np.argwhere(good_indices).squeeze(-1)
# max_cover = segss[chosen_slices][good_indices].max() 
# chosen_idx = (segss == max_cover).argmax()
# diffs[:, good_indices]
# d[good_indices]

chosen_slice = 158
# img size: 154, 241
chosen_crop = [slice(50, 110), slice(60, 125)]
cimg = ct_img[0, 0, :, :, chosen_slice]
clabel = input_img[..., chosen_slice]
cimages = {k: v["pred"][..., chosen_slice] for k, v in images.items()}

img_size = cimages[chosen_method].shape
print(img_size)
imgfolder="plots"

methods = [chosen_method]
methods = [chosen_method] + baselines
methods = [chosen_method, "unet-monai-baseline", "unet-tr-baseline", "unet-dy-baseline"]

n_methods = len(methods)
n_figs = 2 + n_methods
fig_width = 3.5
fig_height = 3
nrows = 1
ncols = n_figs
fontsize=20
plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*fig_width, nrows * fig_height))
fig, ax = plt.subplots(nrows, ncols, figsize=(18, 6))

# ax.subplot(1, n_figs, 1)
ax[0].set_title("CT Image", fontsize=fontsize)
ax[0].imshow(cimg, cmap="gray")
startx, starty = chosen_crop[1].start, img_size[0] - chosen_crop[0].stop
h, w = chosen_crop[0].stop - chosen_crop[0].start, chosen_crop[1].stop - chosen_crop[1].start
print((startx, starty), h, w)
ax[0].add_patch(Rectangle((startx, starty), h, w,
    edgecolor='red',
    facecolor='none',
    lw=2.5))
ax[0].set_xticks([])
ax[0].set_yticks([])

extent=[10, 20, 10, 20]
# ax[1].subplot(1, n_figs, 2)
ax[1].set_title("Label", fontsize=fontsize)
cmap = ListedColormap([[0, 0, 0, 0.0]] + sns.color_palette())
ax[1].imshow(cimg[chosen_crop[0], chosen_crop[1]], cmap="gray")
ax[1].imshow(clabel[chosen_crop[0], chosen_crop[1]], cmap=cmap, alpha=0.8)
ax[1].set_xticks([])
ax[1].set_yticks([])

for ii, img_method in enumerate(methods):
    # ax[0].subplot(1, n_figs, 3)
    ax[2 + ii].set_title(legend[img_method], fontsize=fontsize)
    ax[2 + ii].imshow(cimg[chosen_crop[0], chosen_crop[1]], cmap="gray")
    ax[2 + ii].imshow(cimages[img_method][chosen_crop[0], chosen_crop[1]], cmap=cmap, alpha=0.8)
    ax[2 + ii].set_xlabel(f"{ndsc_mets[img_method] * 100:.2f}", fontsize=fontsize)
    ax[2 + ii].set_xticks([])
    ax[2 + ii].set_yticks([])

plt.tight_layout()
file_location = os.path.join(imgfolder, 'shifts_seg.png')
plt.savefig(file_location, bbox_inches="tight", dpi=200)
print(f"Saving plot to {file_location}")
file_location = os.path.join(imgfolder, 'shifts_seg.pdf')
plt.savefig(file_location, bbox_inches="tight", dpi=200)
print(f"Saving plot to {file_location}")
plt.close()