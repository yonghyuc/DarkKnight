import cv2
import os

import base64
import numpy as np
import torch
from torchvision.ops import nms

from app.model.config import cfg
from app.model.test import im_detect
from app.nets.resnet_v1 import resnetv1

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
demonet = 'res101' # Network to use [vgg16 res101]
dataset = 'pascal_voc_0712' # Trained dataset [pascal_voc pascal_voc_0712]
NMS_THRESH = 0.3
CONF_THRESH = 0.8
img_name = "../data/demo/111/0000000125.png"
target_classes = [6, 7, 15]


def load_module():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    saved_model = os.path.join('app', 'resources', NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    net = resnetv1(num_layers=101)
    net.create_architecture(21, tag='default', anchor_scales=[8, 16, 32])
    net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    return net


def _get_valid_boxes(scores, boxes, category, offset=0):
    idx_boxes = boxes[:, 4*category:4*(category + 1)]
    idx_scores = scores[:, category]
    clean_idx = nms(torch.from_numpy(idx_boxes), torch.from_numpy(idx_scores), NMS_THRESH)
    valid_idxs = clean_idx.numpy()[np.where(idx_scores[clean_idx.numpy()] > CONF_THRESH)]
    valid_boxes = idx_boxes[valid_idxs]
    valid_boxes[:, 0] += offset
    valid_boxes[:, 2] += offset
    return valid_boxes


def get_boxes(img_base64):
    categories = [6, 7, 15]

    nparr = np.fromstring(base64.b64decode(img_base64), np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    mid_x = im.shape[1]//2

    left_scores, left_boxes = im_detect(net, im[:, :mid_x])
    right_scores, right_boxes = im_detect(net, im[:, mid_x:])

    output = dict()
    for category in categories:
        left_valid_boxes = _get_valid_boxes(left_scores, left_boxes, category)
        right_valid_boxes = _get_valid_boxes(right_scores, right_boxes, category, mid_x)
        output[category] = np.concatenate((left_valid_boxes, right_valid_boxes)).tolist()

    return output


net = load_module()

