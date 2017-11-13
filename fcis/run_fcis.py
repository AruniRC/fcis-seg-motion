# ------------------------------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haochen Zhang, Yi Li, Haozhi Qi
#
#   Modified for use on Motion segmentation datasets at UMass Amherst (ARC).
# ------------------------------------------------------------------------------

import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fcis/cfgs/fcis_coco_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
print "use mxnet at", mx.__file__
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_masks import show_masks
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting

#### EDITS ####
import argparse
import utils.image as image
import matplotlib.pyplot as plt
import random
import scipy
import scipy.io as sio
from skimage import color
from skimage.io import imsave

IMAGE_PATH = 'cars2_02.jpg'
IMAGE_DIR = '/data/arunirc/Research/dense-crf-data/training_subset/cars2/'
OUT_PATH = '/data2/arunirc/Research/fcis-seg/FCIS/demo/cars2_02_seg-mask.png'
OUT_DIR = '/data/arunirc/Research/dense-crf-data/fcis-seg-output-sample/'

# OBJ_THRESH = 0.5 # complex background
OBJ_THRESH = 0.1 # camouflaged animal


# ------------------------------------------------------------------------------
def process_masks(im, im_path, opts, detections, masks, class_names, cfg, scale=1.0):
# ------------------------------------------------------------------------------
    """
        Process and save segmentation masks.
        Optionally visualize all detections in one image.
        :param im: [b=1 c h w] in rgb
        :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        :param class_names: list of names in imdb
        :param scale: visualize the scaled image
        :return:
    """

    # --------------------------------------------------------------------------
    # Create output paths
    #   - outputs of each image is saved under a folder with that image's name
    # --------------------------------------------------------------------------
    im_dir, im_fn = os.path.split(im_path)
    _, dir_name = os.path.split(im_dir)
    im_out_dir = im_fn.split('.')[0]  
    out_location =  os.path.join(opts.out_dir, dir_name, im_out_dir)
    if not os.path.isdir(out_location):
            os.makedirs(out_location)


    # --------------------------------------------------------------------------
    #   Iterate over detected object masks
    # --------------------------------------------------------------------------
    obj_scores = []
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        dets = detections[j]
        msks = masks[j]

        obj_count = 0

        for det, msk in zip(dets, msks):

            color = (random.random(), random.random(), random.random())  # generate a random color
            bbox = det[:4] * scale
            cod = bbox.astype(int)

            if im[cod[1]:cod[3], cod[0]:cod[2], 0].size > 0:

                obj_count += 1

                msk = cv2.resize(msk, im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, 0].T.shape)
                bimsk = msk >= cfg.BINARY_THRESH
                bimsk = bimsk.astype(int)
                bimsk = np.repeat(bimsk[:, :, np.newaxis], 3, axis=2)
                mskd = im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] * bimsk

                clmsk = np.ones(bimsk.shape) * bimsk
                clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
                clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
                clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
                im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] = im[cod[1]:cod[3]+1, cod[0]:cod[2]+1, :] + 0.8 * clmsk - 0.8 * mskd
                

                # --------------------------------------------------------------
                # Outputs for each image:
                #   1. Smooth probability mask [imW x imH]
                #   2. Binary object mask [imW x imH]
                #   3. Object score [scalar]
                # --------------------------------------------------------------
                prob_mask = np.zeros((im.shape[0], im.shape[1]))
                prob_mask[cod[1]:cod[3]+1, cod[0]:cod[2]+1] = msk
                bin_mask = prob_mask > 0.5
                obj_score = [det[-1]]
                obj_scores.append(det[-1])


                # --------------------------------------------------------------
                # Save the segmentation outputs under the `out_location` for the image
                #  -- saved as matlab MATLAB-style .mat file
                # --------------------------------------------------------------
                sio.savemat(os.path.join(out_location, str(obj_count).zfill(5) + '_obj-prob.mat'), \
                            dict(objectProb=prob_mask))
                sio.savemat(os.path.join(out_location, str(obj_count).zfill(5) + '_binary-mask.mat'), \
                            dict(objectMask=bin_mask))
                np.savetxt(os.path.join(out_location, str(obj_count).zfill(5) + '_obj-score.txt'), \
                            obj_score)

                if opts.viz:
                    scipy.misc.imsave(os.path.join(out_location, str(obj_count).zfill(5) \
                                                   + '_obj-prob.png'), \
                                                    prob_mask)
                    scipy.misc.imsave(os.path.join(out_location, str(obj_count).zfill(5) \
                                                   + '_binary-mask.png'), \
                                                    bin_mask)

            score = det[-1]
    

    np.savetxt(os.path.join(out_location, 'det_scores.txt'), obj_scores)

    # save the annotated output image
    if opts.viz:
        imsave(os.path.join(opts.out_dir, dir_name, im_fn), im)

    return im


# ------------------------------------------------------------------------------
def parse_input_opts():
# ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Run dense-crf on single frame motion segmentations')
    parser.add_argument('-i', '--image', help='Specify path to RGB image', \
                            default=IMAGE_PATH)
    parser.add_argument('-o', '--out', help='Specify output path for FCIS segmentation', \
                            default=OUT_PATH)
    parser.add_argument('-id', '--img_dir', help='Specify path to folder of RGB images', \
                            default='')
    parser.add_argument('-od', '--out_dir', help='Specify output folder for FCIS segmentation', \
                            default='')
    parser.add_argument('-v', '--viz', help='Save visualizations as images', \
                            default=False, action='store_true')
    opts = parser.parse_args()
    return opts


# ------------------------------------------------------------------------------
def main(opts):
# ------------------------------------------------------------------------------
    # get symbol
    ctx_id = [int(i) for i in config.gpus.split(',')]
    pprint.pprint(config)
    sym_instance = eval(config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 81
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


    # --------------------------------------------------------------------------
    #   Load data
    # --------------------------------------------------------------------------
    if not opts.img_dir:
        # input is a single image, not a folder containing many images
        image_names = [opts.image]
    else:
        # input is a folder of images
        image_names = [ x for x in sorted(os.listdir(opts.img_dir)) \
                        if x.endswith(tuple(['.jpg', '.png', '.JPG', '.PNG'])) ]

    data = []
    for im_name in image_names:
        
        im_path = os.path.join(opts.img_dir, im_name)        
        assert os.path.exists(im_path), ('%s does not exist'.format(im_path))
        im = cv2.imread(im_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})


    # --------------------------------------------------------------------------
    #   Get predictor
    # --------------------------------------------------------------------------
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + '/../model/fcis_coco', 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(ctx_id[0])], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # warm up
    for i in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        _, _, _, _ = im_detect(predictor, data_batch, data_names, scales, config)


    # --------------------------------------------------------------------------
    #   Test
    # --------------------------------------------------------------------------
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

        # ----------------------------------------------------------------------
        #   Bounding-box NMS
        # ----------------------------------------------------------------------
        if not config.TEST.USE_MASK_MERGE:
            print "Mask merge"
            all_boxes = [[] for _ in xrange(num_classes)]
            all_masks = [[] for _ in xrange(num_classes)]
            nms = py_nms_wrapper(config.TEST.NMS)
            for j in range(1, num_classes):
                indexes = np.where(scores[0][:, j] > 0.7)[0]
                cls_scores = scores[0][indexes, j, np.newaxis]
                cls_masks = masks[0][indexes, 1, :, :]
                try:
                    if config.CLASS_AGNOSTIC:
                        cls_boxes = boxes[0][indexes, :]
                    else:
                        raise Exception()
                except:
                    cls_boxes = boxes[0][indexes, j * 4:(j + 1) * 4]

                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j] = cls_dets[keep, :]
                all_masks[j] = cls_masks[keep, :]
            dets = [all_boxes[j] for j in range(1, num_classes)]
            masks = [all_masks[j] for j in range(1, num_classes)]
        else:
            print "GPU mask voting"
            masks = masks[0][:, 1:, :, :]
            im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
            im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
            print (im_height, im_width)
            boxes = clip_boxes(boxes[0], (im_height, im_width))
            result_masks, result_dets = gpu_mask_voting(masks, boxes, scores[0], num_classes,
                                                        100, im_width, im_height,
                                                        config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                        config.BINARY_THRESH, ctx_id[0])

            dets = [result_dets[j] for j in range(1, num_classes)]
            masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]
        print 'testing {} {:.4f}s'.format(im_name, toc())

        
        # ----------------------------------------------------------------------
        #   Thresholding detection confidence
        # ----------------------------------------------------------------------
        for i in xrange(len(dets)):
            keep = np.where(dets[i][:,-1] > OBJ_THRESH)
            dets[i] = dets[i][keep]
            masks[i] = masks[i][keep]

        im_path = os.path.join(opts.img_dir, im_name)
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # ----------------------------------------------------------------------
        #   Process and save predictions
        # ----------------------------------------------------------------------
        process_masks(im, im_path, opts, dets, masks, classes, config)

    print 'done'


if __name__ == '__main__':
    opts = parse_input_opts()
    main(opts)
