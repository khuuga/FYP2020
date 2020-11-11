import os
from os.path import join
import time
from options.train_options import TrainOptions, TestOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm
import cv2
from skimage import io
from skimage import img_as_ubyte

from fusion_dataset import Fusion_Testing_Dataset
from util import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', True)

from google.colab.patches import cv2_imshow

#img_name_list = ['8b01a894fb91025a1dc77611838e9d13', '925f179ba736e15e89bfd6d88e0bba56', '574100c6e31a1dcd096476eb2d632b3c', 'e9b6826aa623549ec77bbc0275002779']
#img_name_list =['000000022969', '000000023781', '000000046872', '000000050145']
img_name_list = [os.path.splitext(filename)[0] for filename in os.listdir('results/')]

show_index = 0
for show_index in range(len(img_name_list)):
    img = cv2.imread('example/'+img_name_list[show_index]+'.png')
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_image)

    img = cv2.imread('results/'+img_name_list[show_index]+'.png')
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, a_pred, b_pred = cv2.split(lab_image)
    a_pred = cv2.resize(a_pred, (l_channel.shape[1], l_channel.shape[0]))
    b_pred = cv2.resize(b_pred, (l_channel.shape[1], l_channel.shape[0]))
    gray_color = np.ones_like(a_pred) * 128

    gray_image = cv2.cvtColor(np.stack([l_channel, gray_color, gray_color], 2), cv2.COLOR_LAB2BGR)
    color_image = cv2.cvtColor(np.stack([l_channel, a_pred, b_pred], 2), cv2.COLOR_LAB2BGR)
    #full_image = np.concatenate([gray_image, color_image], 1)
    save_img_path = 'results_origin/'
    if os.path.isdir(save_img_path) is False:
        print('Create path: {0}'.format(save_img_path))
        os.makedirs(save_img_path)

    cv2.imwrite('results_origin/'+img_name_list[show_index]+'.png', color_image)
