import glob
import os
import sys
import numpy as np
import time
from scipy import ndimage
import SimpleITK as sitk

EPSILON = 1e-32

def compute_binary_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred) + EPSILON
    union = np.sum(y_true) + np.sum(y_pred) - intersection + EPSILON
    iou = intersection / union
    return iou

def evaluation_metrics(label, pred, refine=False):

    iou = compute_binary_iou(label, pred)
    pre = (pred * label).sum() / pred.sum()

    return iou, pre


# =============================== MAIN ========================================
# this is the prediction
submit_dir = os.path.join(sys.argv[1], 'res')
# define the path for gt
gt_dir = os.path.join(sys.argv[1], 'ref')

if not os.path.isdir(submit_dir):
    print("submission dir {} does not exist!".format(submit_dir))
    sys.exit()
if not os.path.isdir(gt_dir):
    print("ground truth dir {} does not exist!".format(gt_dir))
    sys.exit()

# create output
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Calculate metrics
Pre_list, IoU_list, Overall_list = [],[],[]
fids = sorted(glob.glob(gt_dir+'/*.nii.gz'))

for gt_fid in fids:
    print("Start processing ", gt_fid)
    pred_fid = os.path.join(submit_dir, os.path.basename(gt_fid))
    if not os.path.exists(pred_fid):
        raise ValueError("Submission file {} not found, Please check your submission files "
                         "and carefully read the submission guidance.".format(gt_fid))
    print("Find matched gt {} and prediction {} files".format(gt_fid, pred_fid))
    gt_array = sitk.GetArrayFromImage(sitk.ReadImage(gt_fid))
    pred_array = sitk.GetArrayFromImage(sitk.ReadImage(pred_fid))
    assert gt_array.shape == pred_array.shape, "The shape of input array and output array must be the same!"
    assert len(np.unique(pred_array)) == 2, "Please check your predictions, the predictions must be binary (0,1)!"
    iou, pre = evaluation_metrics(gt_array, pred_array)
    IoU_list.append(iou)
    Pre_list.append(pre)
    Overall_list.append((iou + pre) * 0.5)
Metric_list = ['AW_overall', 'AW_IoU', 'AW_Precision']
overall_metric = [np.mean(Overall_list), np.mean(IoU_list), np.mean(Pre_list)]
for i in range(len(Metric_list)):
    print("{}: {:.4f}\n".format(Metric_list[i], float(overall_metric[i])))

output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'w')
for i in range(len(Metric_list)):
    output_file.write("{}: {:.4f}\n".format(Metric_list[i], float(overall_metric[i])))
output_file.close()


