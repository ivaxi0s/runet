# Multiple Sclerosis White Matter Lesions Segmentation

This repo contains code related to Track 1 of the [Shifts Challenge](https://shifts.grand-challenge.org/shifts/). Track 1 is focused on the segmentation of white matter multiple sclerosis lesions on 3D FLAIR magnetic resonance imaging scans.

## Task description


White matter lesions (WML) are regions of inflammation in the brain and are one of the main biomarkers in multiple sclerosis patients. Detection and accurate delineation of WML are important for both diagnosis and prognosis of MS, however manual segmentation is time-consuming and expert-dependent. In the current track, we propose you explore the possibility of automatic WML segmentation from 3D FLAIR scans. Moreover, we allow working on the development of the model's uncertainty measures, which should serve as a proxy for your model's reliability.

Already preprocessed **data can be downloaded** from [zenodo](https://zenodo.org/record/7051658) under the OFSEP data usage agreement. Data is distributed under CC BY NC SA 4.0 license. 

For a more **detailed description of the data** and baseline **experiments**, please refer to our [Datasets & Benchmarks paper](https://arxiv.org/pdf/2206.15407).

For more **practical information** about the organization, data, baseline and submission, please visit our [Grand Challenge](https://shifts.grand-challenge.org/shifts/) web page.
 
## Files description


`metrics.py` is a module containing implementations of metrics used for validation during training and evaluation: Dice score, normalized Dice score (nDSC), lesion F1 score, the area under error retention curve (nDSC R-AUC). Here, nDSC and nDSC R-AUC will be used for your model's and measure's evaluation and are displayed in the [leaderboard](https://shifts.grand-challenge.org/evaluation/ms-lesion-segmentation-phase-i/leaderboard/).

`uncertainty.py` is a module containing implementations of uncertainty measures computed based on deep ensembles: mutual information (MI), expected pair-wise KL divergence (EPKL) and reverse mutual information (RMI) for knowledge uncertainty; expected entropy (ExE) for data uncertainty; entropy of expected (EoE) and negated confidence (NC) for total uncertainty.

`data_load.py` is a module containing implementations of transforms and dataloaders needed for training, validation and inference of a baseline model. Please, go through it carefully if you are not familiar with how to handle MRI data.

`train.py`, `test.py`, `inference.py`, `retention_curves.py` are programs used for reproducing the baseline model. 

## Reproduce the baseline

As the baseline model, a deep ensemble of 3 UNET models was chosen based on the nDSC R-AUC value.
You can download the baseline models from this [link](https://drive.google.com/file/d/1eTTgga7Cd1GjR0YupVbLuLd3unl6_Jj3/view?usp=sharing).


1. Training.

Use the following bash script to sequentially fit models in the ensemble.

```bash
#!/bin/bash
for seed in 1 2 3
do
	python mswml/train.py \
	--seed $seed \
	--path_train_data /path/to/train/FLAIR \
	--path_train_gts /path/to/train/ground/truth/masks \
	--path_val_data /path/to/val/FLAIR \
	--path_val_gts /path/to/val/ground/truth/masks \
	--path_save "/path/to/baselines/dir/seed${seed}"
done
```

2. Evaluation.

Compute performance metrics (nDSC, lesion F1 score, nDSC R-AAC) for an ensemble of models. Metrics are displayed in the terminal.

```bash
python mswml/test.py \
--path_model /path/to/baselines/dir/ \
--path_data /path/to/test/FLAIR \
--path_gts /path/to/test/ground/truth/masks \
--path_bm /path/to/test/fg_masks \
--threshold 0.35
```

Additional parameters like `--num_workers` and `--n_jobs` control the number of workers used for parallel processing of images and parallel computation of lesion F1 score respectively.

The probability threshold `threshold` is used for obtaining binary lesion masks from probability output.

3. Inference.

Perform inference for an ensemble of baseline models and save 3D Nifti images of predicted probability maps averaged across ensemble models (saved to "*pred_prob.nii.gz" files), binary segmentation maps predicted obtained by thresholding of average predictions and removing all connected components smaller than 9 voxels (saved to "*pred_seg.nii.gz"), uncertainty maps for reversed mutual information measure (saved to "*uncs_rmi.nii.gz").

```bash
python mswml/inference.py \
--path_pred /path/to/dir/to/save/predictions/ \
--path_model /path/to/baselines/dir/ \
--path_data /path/to/test/FLAIR \
--path_bm /path/to/test/fg_masks \
--threshold 0.35
```

## Requirements

We used python version 3.8.10. For all additional library requirements, please refer to 
`requirements.txt` file.

## Data visualization

Both 3D FLAIR images and ground truth segmentation masks are distributed in 
[Nifti](http://nifti.nimh.nih.gov) format.

For the visualization of images, we suggest either using [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) software 
for medical images visualization or [nilearn](https://nilearn.github.io/stable/index.html) 
Python library for displaying cuts of 3D images. For the last option consider 
using `nilearn.plotting.plot_img()` function to display slices of a 
3D image and `nilearn.plotting.plot_roi()` for displaying slices of a 3D image 
with additional overlays, e.g. ground truth masks or predicted binary masks.

