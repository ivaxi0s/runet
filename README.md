# Model Overview
This repository contains the code for RUNet -  Relational UNet for Image
Segmentation. 
The code presents a volumetric (3D) multi-organ segmentation application using the BTCV challenge dataset.
![image](https://lh3.googleusercontent.com/pw/AM-JKLU2eTW17rYtCmiZP3WWC-U1HCPOHwLe6pxOfJXwv2W-00aHfsNy7jeGV1dwUq0PXFOtkqasQ2Vyhcu6xkKsPzy3wx7O6yGOTJ7ZzA01S6LSh8szbjNLfpbuGgMe6ClpiS61KGvqu71xXFnNcyvJNFjN=w1448-h496-no?authuser=0)

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Training

```./pretrained_models```

The following command initiates finetuning using the pretrained checkpoint:
``` bash
python main.py
--batch_size=1
--logdir=unetr_pretrained
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5
--save_checkpoint
--data_dir=/dataset/dataset0/
--pretrained_dir='./pretrained_models/'
--model=unet-attn-rsa

--resume_ckpt
```
Change ``model`` to train with different architechures.


### Testing

The following command runs inference using the checkpoint:
``` bash
python test.py
--infer_overlap=0.5
--data_dir=/dataset/dataset0/
--pretrained_dir='./pretrained_models/'
--saved_checkpoint=ckpt
--model=unet-attn-rsa
```

## Dataset
![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0)

The training data is from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).

- Target: 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kideny 4.Gallbladder 5.Esophagus 6. Liver 7. Stomach 8.Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12.Right adrenal gland 13.Left adrenal gland.
- Task: Segmentation
- Modality: CT
- Size: 30 3D volumes (24 Training + 6 Testing)


## Citation
This repo was built from
```
@inproceedings{hatamizadeh2022unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={574--584},
  year={2022}
}
```

## References
[1] Hatamizadeh, Ali, et al. "UNETR: Transformers for 3D Medical Image Segmentation", 2021. https://arxiv.org/abs/2103.10504.

[2] Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
", 2020. https://arxiv.org/abs/2010.11929.
