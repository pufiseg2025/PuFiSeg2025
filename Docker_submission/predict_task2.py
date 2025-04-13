#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK as sitk
import os

input_dir = '/input/'
path_img = os.path.join(input_dir,'{}.nii.gz')
path_pred = '/output/{}.nii.gz'

list_case = [k.split('.')[0] for k in os.listdir(input_dir)]

for case in list_case:
    img = sitk.ReadImage(path_img.format(case))

    ##
    # =======your logic here. Below we do binary thresholding as a demo=====

    # using SimpleITK to do binary thresholding between 100 - 10000
    vs_pred = your model(img)

    result = postprocess(vs_pred)
    # ======================================================================
    # please make sure the results were processed with largest component extraction
    result = sitk.GetImagefromArray(result)
    # result.CopyInformation(img)
    # save the segmentation mask
    sitk.WriteImage(result, path_pred.format(case))
