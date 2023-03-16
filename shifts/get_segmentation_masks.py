"""
Computation of performance metrics (nDSC, lesion F1 score, nDSC R-AUC)
for an ensemble of models.
Metrics are displayed in console.
"""

import argparse
import glob
import os
import torch
from monai.inferers import sliding_window_inference
import numpy as np
from tqdm import tqdm
from data_load import remove_connected_components, get_val_dataloader
from metrics import dice_norm_metric
from model_utils import create_model

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

    chosen_idx = 21 
    ''' Evaluation loop '''
    with torch.no_grad():
        for count, batch_data in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing"):
            if count != chosen_idx:
                continue
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

            # compute metrics
            ndsc += [dice_norm_metric(ground_truth=gt, predictions=seg)]

            if count == chosen_idx:
                tqdm.write(f"Saving prediction with ndsc {ndsc[-1] * 100:.2f} to disk.")
                parts = np.array([dice_norm_metric(ground_truth=gt[..., i], predictions=seg[..., i]) for i in range(seg.shape[-1])])
                for f in files:
                    head, tail = os.path.split(f)
                    save_dir = os.path.join(head, "images")
                    os.makedirs(save_dir, exist_ok=True)
                    fname = os.path.join(save_dir, f"{args.path_data_name}_pred.npz")
                    np.savez_compressed(fname, image=batch_data["image"].cpu().numpy(), gt=gt, pred=seg)



    # ndsc = np.asarray(ndsc) * 100.
    # f1 = np.asarray(f1) * 100.
    # ndsc_aac = np.asarray(ndsc_aac) * 100.

    # tqdm.write(f"nDSC:\t{np.mean(ndsc):.4f} +- {np.std(ndsc):.4f}")
    # tqdm.write(f"Lesion F1 score:\t{np.mean(f1):.4f} +- {np.std(f1):.4f}")
    # tqdm.write(f"nDSC R-AUC:\t{np.mean(ndsc_aac):.4f} +- {np.std(ndsc_aac):.4f}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


# def calculate_metrics(metric_name):
#     met_dd = defaultdict()
#     files = sorted(glob.glob(f"logs/**/**/{metric_name}_scores.npz"))
#     for fname in files:
#         parts = Path(fname).parts
#         run_name = parts[-3]
#         x = np.load(fname)
#         met_dd[run_name] = x["ndsc"]
#     return met_dd

# from pprint import pprint
# import numpy as np

# chosen_method="unet-attn-rsa"
# model_names = ["unet-monai", "unet-tr", "unet-res", "unet-attn-rsa", "unet-dy", "unet-3d"]
# model_names = [f"{x}-baseline" for x in model_names]

# baselines = [x for x in model_names if chosen_method not in x]

# dev_out = calculate_metrics("dev_out")
# dev_out = {k: v for k, v in dev_out.items() if k in model_names}
# highs = {k: [v.max(), v.argmax()] for k, v in dev_out.items()} 
# lows = {k: [v.min(), v.argmin()] for k, v in dev_out.items()} 

# diffs = np.array([dev_out["unet-attn-rsa-baseline"] - dev_out[b] for b in baselines])

# pprint(highs)
# pprint(lows)

# data = np.array(list(dev_out.values()))
# good_choices = np.argwhere((diffs > 0).mean(0) == 1).squeeze(-1)
# diffs[:, good_choices].mean(0)
# good_choices[-2] # Image 21  is a good candidate