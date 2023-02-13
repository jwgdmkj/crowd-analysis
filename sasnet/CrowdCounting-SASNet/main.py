# Copyright 2021 Tencent

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# https://github.com/TencentYoutuResearch/CrowdCounting-SASNet
# 목표 : gt_map같은거 없어도 돌아가도록 개조하기
# python3 main.py --data_path ./data/part_B_sample --model_path ./weights/SHHB.pth
# python3 main.py --data_path ./data/ShanghaiTech/part_B_final --model_path ./weights/SHHB.pth

import os
import numpy as np
import torch
import argparse
from model import SASNet
import warnings
import matplotlib.pyplot as plt
import random
from datasets.loading_data import loading_data
warnings.filterwarnings('ignore')


# define the GPU id to be used
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Inference setting', add_help=False)
    parser.add_argument('--model_path', type=str, help='path of pre-trained model')
    parser.add_argument('--data_path', type=str, help='root path of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--log_para', type=int, default=1000, help='magnify the target density map')
    parser.add_argument('--block_size', type=int, default=32, help='patch size for feature level selection')

    return parser

# get the dataset
def prepare_dataset(args):
    return loading_data(args)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # update the moving average
    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

def pcolormesh(pred_map):
    # Sample data
    Xlen, Ylen = pred_map.shape[3], pred_map.shape[2]
    Xside, Yside = np.linspace(0, Xlen -1, Xlen), np.linspace(0, Ylen - 1, Ylen)
    X, Y = np.meshgrid(Xside, Yside)

    Z = []
    for i in range(Ylen-1, 0, -1):
        tmp = []
        for j in range(Xlen):
            try:
                tmp.append(pred_map[0][0][i][j])
            except:
                print(i, j)

        Z.append(tmp)
    Z = np.array(Z)

    # Plot the density map using nearest-neighbor interpolation
    plt.pcolormesh(X, Y, Z)
    plt.savefig('savefig_sample.png')

# ------------------------------------mine ------------------------------------------#

def main(args):
    """the main process of inference"""
    test_loader = prepare_dataset(args)

    model = SASNet(args=args).cuda()
    # load the trained model
    model.load_state_dict(torch.load(args.model_path))
    print('successfully load model from', args.model_path)

    with torch.no_grad():
        model.eval()

        # iterate over the dataset
        for vi, data in enumerate(test_loader, 0):
            print(vi)
            img = data

            img = img.cuda()
            # get the predicted density map - pred_map이 결과물. 이를 density map으로 표현 필요
            pred_map = model(img)

            pred_map = pred_map.data.cpu().numpy()
            # evaluation over the batch
            for i_img in range(pred_map.shape[0]):
                pred_cnt = np.sum(pred_map[i_img]) / args.log_para

            if vi == 0:
                pcolormesh(pred_map)

# ------------------------------------original ------------------------------------------#
"""
def main(args):
    # the main process of inference
    test_loader = prepare_dataset(args)

    model = SASNet(args=args).cuda()
    # load the trained model
    model.load_state_dict(torch.load(args.model_path))
    print('successfully load model from', args.model_path)

    with torch.no_grad():
        model.eval()

        maes = AverageMeter()
        mses = AverageMeter()
        # iterate over the dataset
        for vi, data in enumerate(test_loader, 0):
            print(vi)
            img, gt_map = data

            img = img.cuda()
            gt_map = gt_map.type(torch.FloatTensor).unsqueeze(0).cuda()
            # get the predicted density map - pred_map이 결과물. 이를 density map으로 표현 필요
            pred_map = model(img)

            pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.data.cpu().numpy()
            # evaluation over the batch
            for i_img in range(pred_map.shape[0]):
                pred_cnt = np.sum(pred_map[i_img]) / args.log_para
                gt_count = np.sum(gt_map[i_img])

                maes.update(abs(gt_count - pred_cnt))
                mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))

            if vi == 0:
                pcolormesh(pred_map)

        # calculation mae and mre
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        # print the results

        print('=' * 50)
        print('    ' + '-' * 20)
        print('    [mae %.3f mse %.3f]' % (mae, mse))
        print('    ' + '-' * 20)
        print('=' * 50)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SASNet inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)