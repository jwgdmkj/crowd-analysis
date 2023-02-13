#!/usr/bin/env python
# coding: utf-8

import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.utils.data as data
from imageio import imsave
from torchvision import transforms
from tqdm import tqdm
from albumentations.pytorch import ToTensor
from head_detection.vision.utils import collate_fn as coco_collate

import tracemalloc

def to_torch(im):
    transf = ToTensor()
    torched_im = transf(image=im)['image'].to(torch.device("cuda"))
    return torch.unsqueeze(torched_im, 0)


def get_state_dict(net, pt_model, only_backbone=False):
    """
    Restore weight. Full or partial depending on `only_backbone`.
    """
    strict = False if only_backbone else True
    state_dict = torch.load(pt_model)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        # Remove Head module while restoring network
        if only_backbone:
            if 'Head' in name.split('.')[0]:
                continue
            else:
                name = 'backbone.' + name
        new_state_dict[name] = v
    return new_state_dict


def restore_network(net, pt_model, only_backbone=False):
    """
    Restore weight. Full or partial depending on `only_backbone`.
    """
    strict = False if only_backbone else True
    state_dict = torch.load(pt_model)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        # Remove Head module while restoring network
        if only_backbone:
            if 'Head' in name.split('.')[0]:
                continue
            else:
                name = 'backbone.' + name
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=strict)
    print('Loaded the entire model in %r mode' %strict)
    return net


def visualize(base_path, test_dataset, plot_dir, batch_size=4, ):
    """Visualize ground truth data"""
    device = torch.device('cuda')
    dataset = HeadDataset(test_dataset,
                          base_path,
                          dataset_param={},
                          train=False)
    batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=coco_collate))
    for ind, (images, targets) in enumerate(tqdm(batch_iterator)):
        images = list(img.to(device) for img in images)
        np_images = [(ims.cpu().numpy()*255.).astype(np.uint8) for ims in images]
        gt_boxes = [gt['boxes'].numpy().astype(np.float64) for gt in targets]
        for np_im, gt_box in zip(np_images, gt_boxes):
            plot_images = plot_ims(np_im, [], gt_box)
            imsave(osp.join(plot_dir, str(ind) + '.jpg'), plot_images)

# --------------------------------- Under is My Code ------------------------------------- #
def get_inner_ratio(curimgpoint, previmgpoint):
    curbox = (curimgpoint[0][0], curimgpoint[0][1], curimgpoint[1][0], curimgpoint[1][1])
    prevbox = (previmgpoint[0][0], previmgpoint[0][1], previmgpoint[1][0], previmgpoint[1][1])
    # print('constast ', curbox, prevbox)

    box1_area = (curbox[2] - curbox[0] + 1) * (curbox[3] - curbox[1] + 1)
    box2_area = (prevbox[2] - prevbox[0] + 1) * (prevbox[3] - prevbox[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(curbox[0], prevbox[0])
    y1 = max(curbox[1], prevbox[1])
    x2 = min(curbox[2], prevbox[2])
    y2 = min(curbox[3], prevbox[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

# crowdbox [0] : 박스 수, [1] : (좌상단좌표), (우하단좌표), 인식횟수
def deadlockcheck(crowdbox_assembler, curimgcrowd) :
    # 고려해야 할 것
    # 1. 기존 것 중 1 추가되어야 하는 것
    # 2. 기존 것 중 제거되어야 하는 것
    # 3. 새 것 중 추가되어야 하는 것
    should_be_delete = []
    should_be_add = []
    # print('\nprev crowd ', crowdbox_assembler)
    # print('now crowd ', curimgcrowd)

    for i in range(len(crowdbox_assembler)):
        should_be_delete.append(True)
    for i in range(len(curimgcrowd)):
        should_be_add.append(True)
    
    for boxid in range(len(crowdbox_assembler)):  # 이전 박스들 중
        for curboxid in range(len(curimgcrowd)): # 현재 박스와 비교해서
            # 겹치는 영역이 0.8 이상이면 갈아끼우고 중복횟수 1 증가
            ratio = get_inner_ratio(curimgcrowd[curboxid], crowdbox_assembler[boxid])
            # print('ratio is ', ratio)
            if ratio > 0.5 :
                crowdbox_assembler[boxid][0] = curimgcrowd[curboxid][0]
                crowdbox_assembler[boxid][1] = curimgcrowd[curboxid][1]
                crowdbox_assembler[boxid][2] += 1
                should_be_delete[boxid] = False   # 새로 값이 갱신된 놈은, 제거될 필요가 없다.
                should_be_add[curboxid] = False   # 기존 있던 놈과 겹치는 놈은, 추가될 필요가 없다.
                # print('so now ', crowdbox_assembler)

    # print(should_be_delete)
    # print(should_be_add)
    for boxid, reverse_tf in reversed(list(enumerate(should_be_delete))) :
        if should_be_delete[boxid] == True :
            crowdbox_assembler.pop(boxid)
            # print('pop, so ', crowdbox_assembler)

    for curboxid in range(len(curimgcrowd)) :
        if should_be_add[curboxid] == True:
            crowdbox_assembler.append([curimgcrowd[curboxid][0], curimgcrowd[curboxid][1], 1])

    # print('now new crowd ', crowdbox_assembler)

    for boxid in range(len(crowdbox_assembler)) :
        if crowdbox_assembler[boxid][2] >= 5 :
            return True, crowdbox_assembler[boxid]

    return False, -1

def dense_dfs(startnode, visit, hajimariX, hajimariY, hateX, hateY, headnum, boxlen_assembler, center_assembler, flag) :
    visit[startnode] = 1

    for nextnode in range(startnode+1, len(boxlen_assembler)):
        # startnode에서 nextnode로 넘어갈 때, nextnode를 방문한 적이 없고
        # startnode와 nextnode 사이의 거리가 startnode의 박스 길이의 2배 이하일 경우, dfs실행
        if visit[nextnode] == 0 and \
                ((center_assembler[startnode][0] - center_assembler[nextnode][0]) ** 2 + \
                (center_assembler[startnode][1] - center_assembler[nextnode][1]) **2 ) ** (1/2) \
                <= 2.5 * boxlen_assembler[startnode]:

            flag = True
            headnum += 1
            hajimariX = center_assembler[startnode][0] if hajimariX > center_assembler[startnode][0] else hajimariX
            hajimariX = center_assembler[nextnode][0] if hajimariX > center_assembler[nextnode][0] else hajimariX
            hajimariY = center_assembler[startnode][1] if hajimariY > center_assembler[startnode][1] else hajimariY
            hajimariY = center_assembler[nextnode][1] if hajimariY > center_assembler[nextnode][1] else hajimariY

            hateX = center_assembler[startnode][0] if hateX < center_assembler[startnode][0] else hateX
            hateX = center_assembler[nextnode][0] if hateX < center_assembler[nextnode][0] else hateX
            hateY = center_assembler[startnode][1] if hateY < center_assembler[startnode][1] else hateY
            hateY = center_assembler[nextnode][1] if hateY < center_assembler[nextnode][1] else hateY

            # print('connected ', startnode, nextnode, end=' ')
            # print(' and now box is ', hajimariX, hajimariY, hateX, hateY, end = ' ')
            # print(' until head is ', headnum)

            hajimariX, hajimariY, hateX, hateY, flag, headnum = \
                dense_dfs(nextnode, visit, hajimariX, hajimariY, hateX, hateY, headnum, boxlen_assembler, center_assembler, flag)
    # print('returned is ', hajimariX, hajimariY, hateX, hateY, flag)
    return hajimariX, hajimariY, hateX, hateY, flag, headnum


# i번째 요소에, 그 상자의 중심좌표와 그 상자의 길이가 들어있다.
def check_density(plotting_im, boxlen_assembler, center_assembler) :
    # 큰 붉은 상자를 그리기 위한, 최좌상좌표 및 최우하좌표
    hajimariX, hajimariY, hateX, hateY = 10000, 10000, -1, -1
    crowdbox = []

    # 우선, i번째 상자와 그 길이 2배 이하인 것들중 가장 가까운 것과 그 거리를 구함.
    # dfs를 한 번 써보겠다.
    visit = []
    for i in range(len(boxlen_assembler)) :
        visit.append(0)

    for idx in range(len(boxlen_assembler)):
        if visit[idx] == 1 : continue   # 이미 방문한 놈은 pass
        # 방문 안한 놈에 한해, dfs 실행
        flag = False
        headnum = 1
        x1, y1, x2, y2, flag, headnum = \
            dense_dfs(idx, visit, hajimariX, hajimariY, hateX, hateY, headnum, boxlen_assembler, center_assembler, flag)
        if flag and headnum >= 10:
            cv2.rectangle(plotting_im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            crowdbox.append([(x1, y1), (x2, y2)])
        hajimariX, hajimariY, hateX, hateY = 10000, 10000, -1, -1
    return crowdbox


# pred_box = [[box1의 id, 그 좌표], [box2의 id, 그 좌표], ... [boxn의 id 및 그 좌표]]
# headnum과 거리계산(2* boxlen_assembler[startnode]) - 언제든 편의로 바꿀 수 있는 값.
def plot_ims(img, pred_box, gt_box=None, text=True):
    """
    Prediction : Yellow
    Ground Truth : Green
    """

    center_assembler = []
    boxlen_assembler = []

    plotting_im = img.transpose(1,2,0).copy()
    if gt_box is None:
        gt_box = []

    for b_id, box in enumerate(gt_box):
        (startX, startY, endX, endY) = [int(i) for i in box]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cur_centroid = tuple([(startX+endX)//2,
                              (startY+endY)//2])
        if text:
            cv2.putText(plotting_im, str(b_id), cur_centroid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 0.5)

    # 찾은 머리 수(len(pred_box)) 만큼 그리기 진행.
    # cur_centroid : 한 box의 중심. 이를 기준으로 각 상자끼리의 거리를 잴 예정.
    for b_id, box in enumerate(pred_box):
        (startX, startY, endX, endY) = [int(i) for i in box]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (255, 255, 0), 2)
        cur_centroid = tuple([(startX+endX)//2,
                              (startY+endY)//2])

        center_assembler.append(cur_centroid)
        boxlen_assembler.append(abs(startX-endX))

        if text:
            cv2.putText(plotting_im, str(b_id), cur_centroid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.putText(plotting_im, 'Detected Head - ' + str(len(pred_box)), tuple([0, plotting_im.shape[0]]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    """ 군중 판단 및 그 정체 시간 탐색 """
    # crowdbox = check_density(plotting_im, boxlen_assembler, center_assembler)
    """ 군중 판단 및 그 정체 시간 탐색 종료 """
    return plotting_im
    # return plotting_im, crowdbox