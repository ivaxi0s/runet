"""
Computation of performance metrics (nDSC, lesion F1 score, nDSC R-AUC)
for an ensemble of models.
Metrics are displayed in console.
"""

import argparse
import glob
import os
import torch
from joblib import Parallel
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import numpy as np
from tqdm import tqdm
from data_load import remove_connected_components, get_val_dataloader
from metrics import dice_norm_metric, lesion_f1_score, ndsc_aac_metric
from model_utils import create_model
from uncertainty import ensemble_uncertainties_classification

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# model
parser.add_argument('--num_models', type=int, default=-1,
                    help='Number of models in ensemble')
parser.add_argument('--path_model', type=str, default='logs',
                    help='Specify the dir to al the trained models')
parser.add_argument('--model', type=str, default="unet-monai", help='Specify the global random seed')
parser.add_argument('--id', type=str, default='baseline',
                    help='Specify the model name')
# data
parser.add_argument('--path_data_name', type=str, required=True,
                    help='Specify the name of split. Used to name the metric file.')
parser.add_argument('--path_data', type=str, required=True,
                    help='Specify the path to the directory with FLAIR images')
parser.add_argument('--path_gts', type=str, required=True,
                    help='Specify the path to the directory with ground truth binary masks')
parser.add_argument('--path_bm', type=str, required=True,
                    help='Specify the path to the directory with brain masks')
# parallel computation
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to preprocess images')
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')

# RSA Params
parser.add_argument('--rsa', action='store_true', help='Enables RSA')
parser.add_argument('--rsa-enc', action='store_true', help='Enables RSA at the Encoder')
parser.add_argument('--rsa-dec', action='store_true', help='Enables RSA at the Decoder')
parser.add_argument('--rsa-first', action='store_true', help='Enables RSA at the first layer of DoubleConv Block')
parser.add_argument('--rsa-second', action='store_true', help='Enables RSA at the second layer of DoubleConv Block')
parser.add_argument('--rsa-pos', nargs='+', default=[1, 2, 3, 4], type=int, help='Define RSA positions')



def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    args.path_data = os.path.expanduser(args.path_data)
    args.path_gts = os.path.expanduser(args.path_gts)
    args.path_bm = os.path.expanduser(args.path_bm)

    '''' Initialise dataloaders '''
    val_loader = get_val_dataloader(flair_path=args.path_data,
                                    gts_path=args.path_gts,
                                    num_workers=args.num_workers,
                                    bm_path=args.path_bm)

    path_save = os.path.join(args.path_model, f"{args.model}-{args.id}")
    print(f"Evaluating with {path_save}")
    files = sorted(glob.glob(os.path.join(path_save, "seed*", "Best_model_finetuning.pth")))


    ''' Load trained models  '''
    K = args.num_models if args.num_models != -1 else len(files)
    models = []
    for i in range(K):
        model = create_model(args)
        model = model.to(device)
        models.append(model)

    print(f"Using model {models[0].__class__}")

    for i, model in enumerate(models):
        model.load_state_dict(torch.load(files[i], map_location=device))
        model.eval()

    act = torch.nn.Softmax(dim=1)
    th = args.threshold
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    ndsc, f1, ndsc_aac = [], [], []

    ''' Evaluation loop '''
    with Parallel(n_jobs=args.n_jobs) as parallel_backend:
        with torch.no_grad():
            for count, batch_data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing"):
                inputs, gt, brain_mask = (
                    batch_data["image"].to(device),
                    batch_data["label"].cpu().numpy(),
                    batch_data["brain_mask"].cpu().numpy()
                )

                # get ensemble predictions
                all_outputs = []
                for model in models:
                    outputs = sliding_window_inference(inputs, roi_size,
                                                       sw_batch_size, model,
                                                       mode='gaussian')
                    outputs = act(outputs).cpu().numpy()
                    outputs = np.squeeze(outputs[0, 1])
                    all_outputs.append(outputs)
                all_outputs = np.asarray(all_outputs)

                # obtain binary segmentation mask
                seg = np.mean(all_outputs, axis=0)
                seg[seg >= th] = 1
                seg[seg < th] = 0
                seg = np.squeeze(seg)
                seg = remove_connected_components(seg)

                gt = np.squeeze(gt)
                brain_mask = np.squeeze(brain_mask)

                # compute reverse mutual information uncertainty map
                uncs_map = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(all_outputs, axis=-1),
                     np.expand_dims(1. - all_outputs, axis=-1)),
                    axis=-1))['reverse_mutual_information']

                # compute metrics
                ndsc += [dice_norm_metric(ground_truth=gt, predictions=seg)]
                f1 += [lesion_f1_score(ground_truth=gt,
                                       predictions=seg,
                                       IoU_threshold=0.5,
                                       parallel_backend=parallel_backend)]
                ndsc_aac += [ndsc_aac_metric(ground_truth=gt[brain_mask == 1].flatten(),
                                             predictions=seg[brain_mask == 1].flatten(),
                                             uncertainties=uncs_map[brain_mask == 1].flatten(),
                                             parallel_backend=parallel_backend)]

                # for nervous people
                # if count % 10 == 0:
                #     print(f"Processed {count}/{len(val_loader)}")

    ndsc = np.asarray(ndsc) * 100.
    f1 = np.asarray(f1) * 100.
    ndsc_aac = np.asarray(ndsc_aac) * 100.

    tqdm.write(f"nDSC:\t{np.mean(ndsc):.4f} +- {np.std(ndsc):.4f}")
    tqdm.write(f"Lesion F1 score:\t{np.mean(f1):.4f} +- {np.std(f1):.4f}")
    tqdm.write(f"nDSC R-AUC:\t{np.mean(ndsc_aac):.4f} +- {np.std(ndsc_aac):.4f}")

    for f in files:
        head, tail = os.path.split(f)
        fname = os.path.join(head, f"{args.path_data_name}_scores.npz")
        np.savez_compressed(fname, ndsc=ndsc, fi=f1, ndsc_aac=ndsc_aac)


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
