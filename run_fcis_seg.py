#!/home/arunirc/dense-crf/bin/python


'''
    Dense CRF Motion Segmentation refinement
    ----------------------------------------
    - Specify data locations and settings below. 
    - Alternatively, you can call this script from the cmd line and pass the args:
        > python run_fcis_seg.py -i IMAGE_DATASET -o OUTPUT_LOCATION -d DATASET_NAME
    - Optional: modify path to Python interpreter in the first line of this script.
'''

from __future__ import division


import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from skimage import color
from skimage.io import imread, imsave
import os
from os import listdir, makedirs
from os.path import isfile, join, isdir
import argparse
import subprocess
import shutil


# IMAGE_DATA = '/data2/arunirc/Research/FlowNet2/flownet2-docker/data/complexBackground/complexBackground-multilabel/'
# IMAGE_DATA = '/data2/arunirc/Research/FlowNet2/flownet2-docker/data/CamAnimal/CamouflagedAnimalDataset/'
IMAGE_DATA = '/data2/arunirc/Research/FlowNet2/flownet2-docker/data/FBMS/Trainingset/'
IMAGE_EXT = ['.jpg', '.png', '.JPG', '.PNG']
OUT_DIR = 'data/fcis-output/FBMS-Train'


def parse_input_opts():
    parser = argparse.ArgumentParser(description='Visualize flow')
    parser.add_argument('-i', '--image_data', help='Specify folder containing RGB dataset', \
                            default=IMAGE_DATA)
    parser.add_argument('-o', '--out_dir', help='Specify output folder for CRF segmentaitons', \
                            default=OUT_DIR)
    parser.add_argument('-d', '--dataset', help='Specify dataset: davis, camo, complex, fbms', \
                            default='complex')
    opts = parser.parse_args()
    opts.image_exts = IMAGE_EXT
    
    return opts


# ------------------------------------------------------------------------------
def apply_fcis_seg(opts):
# ------------------------------------------------------------------------------

    for d in sorted(listdir(opts.image_data)):

        # FBMS videos have inconsistent numbering for frames
        MARPLE_FLAG = False 
        TENNIS_FLAG = False
        
        vid_dir = join(opts.image_data, d)
        if not isdir(vid_dir):
            continue

        vid_out_dir = opts.out_dir

        print join(vid_dir)


        # ----------------------------------------------------------------------
        # Dataset specific hackery
        # ----------------------------------------------------------------------
        if opts.dataset == 'davis':
            pass

        elif opts.dataset == 'complex':
            pass

        elif opts.dataset == 'camo':
            vid_dir = join(vid_dir, 'frames/')
            vid_out_dir = join(opts.out_dir, d)

        elif opts.dataset == 'fbms':
            pass

        cmd = 'python ./fcis/run_fcis.py ' \
                        + '-id ' + vid_dir + ' ' \
                        + '-od ' + vid_out_dir + ' -v'
        print cmd
        subprocess.call(cmd, shell=True)


# entry point
if __name__ == '__main__':
    opts = parse_input_opts()
    apply_fcis_seg(opts)
    