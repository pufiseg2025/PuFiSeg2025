#pip install lpips
#pip install git+https://github.com/dingkeyan93/DISTS.git
#pip install SimpleITK scikit-image tqdm

import os
import sys
import glob
import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from scipy import linalg
import torch
import lpips
from tqdm import tqdm
import json


EPSILON = 1e-8


def extract_features(volume):
    return volume.reshape(volume.shape[0], -1)

def normalize_volume(vol, min_val=-1024, max_val=3071):
    return (vol - min_val) / (max_val - min_val + EPSILON)


# main
submit_dir = os.path.join(sys.argv[1])
gt_dir = os.path.join(sys.argv[2])
output_dir = sys.argv[3]


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

lpips_model = lpips.LPIPS(net='vgg')


psnr_list, ssim_list, lpips_list = [], [], []

fids = sorted(glob.glob(gt_dir + "/*.nii.gz"))[40:]
print(f"Found {len(fids)} files in {gt_dir}")
for gt_path in tqdm(fids):
    fname = os.path.basename(gt_path)
    pred_path = os.path.join(submit_dir, 'LD_'+fname)

    if not os.path.exists(pred_path):
        print(pred_path)
        raise FileNotFoundError(f"Prediction file {fname} not found in submission.")

    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(np.float32)
    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).astype(np.float32)

    if gt_img.shape != pred_img.shape:
        raise ValueError(f"Shape mismatch: {fname}")

    psnr_vol, ssim_vol, lpips_vol, dists_vol = [], [], [], []

    for i in range(gt_img.shape[0]):
        print(f"Processing slice {i+1}/{gt_img.shape[0]} in {fname}")
        gt_slice = normalize_volume(gt_img[i])
        pred_slice = normalize_volume(pred_img[i])

        psnr_val = compute_psnr(gt_slice, pred_slice, data_range=1.0)
        ssim_val = compute_ssim(gt_slice, pred_slice, data_range=1.0)

        gt_tensor = torch.from_numpy(gt_slice[None, None, :, :]).repeat(1, 3, 1, 1).float()
        pred_tensor = torch.from_numpy(pred_slice[None, None, :, :]).repeat(1, 3, 1, 1).float()

        lpips_val = lpips_model(gt_tensor, pred_tensor).item()
  

        psnr_vol.append(psnr_val)
        ssim_vol.append(ssim_val)
        lpips_vol.append(lpips_val)
  

    psnr_list.append(np.mean(psnr_vol))
    ssim_list.append(np.mean(ssim_vol))
    lpips_list.append(np.mean(lpips_vol))

psnr_avg = np.mean(psnr_list)
ssim_avg = np.mean(ssim_list)
lpips_avg = np.mean(lpips_list)


# Output the average value of each metric
result_dict = {
    "psnr": round(psnr_avg, 4),
    "ssim": round(ssim_avg, 4),
    "lpips": round(lpips_avg, 4)
}

with open(os.path.join(output_dir, "scores.txt"), "w") as f:
    for k, v in result_dict.items():
        f.write(f"{k}: {v}\n")

with open(os.path.join(output_dir, "result.json"), "w") as f:
    json.dump(result_dict, f)
