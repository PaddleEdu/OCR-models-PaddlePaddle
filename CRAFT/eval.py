"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse
import paddle
import cv2
import numpy as np
import craft_utils
import imgproc
from utils import get_files, saveResult
from x2paddle_code import CRAFT


from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='/home/aistudio/CRAFT/PaddlePretrainedModel/craft_ic15_20k.pdparams', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/home/aistudio/Data/data/ICDAR2015/Challenge4/ch4_test_images/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='/home/aistudio/CRAFT/PaddlePretrainedModel/craft_refiner_CTW1500.pdparams', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = get_files(args.test_folder)

result_folder = './outputs/submit_ic15/'
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = paddle.to_tensor(x).transpose([2, 0, 1])    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                            # [c, h, w] to [b, c, h, w]

    # forward pass
    with paddle.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].numpy()
    score_link = y[0,:,:,1].numpy()

    # refine link
    if refine_net is not None:
        with paddle.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize
    net = paddle.DataParallel(net)

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    # net.set_state_dict(copyStateDict(paddle.load(args.trained_model)))
    net.set_state_dict(paddle.load(args.trained_model))
    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from x2paddle_refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        refine_net.set_state_dict(copyStateDict(paddle.load(args.refiner_model)))
        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        if k == len(image_list)-1:
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
        else:
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.poly, refine_net)

        # save score text
        saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
    print("elapsed time : {}s".format(time.time() - t))
