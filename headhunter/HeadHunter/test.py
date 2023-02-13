#!/usr/bin/env python
# coding: utf-8
import cv2
import argparse
import csv
import os
import os.path as osp
import time
from glob import glob

# from mean_average_precision.detection_map import DetectionMAP
import numpy as np
import torch    #torchvision == 0.7.0
from tqdm import tqdm
from PIL import Image

from head_detection.data import (cfg_mnet, cfg_res50, cfg_res50_4fpn,
                                 cfg_res152, ch_anchors, combined_anchors,
                                 headhunt_anchors, sh_anchors)
from head_detection.models.head_detect import customRCNN
# from head_detection.utils import get_state_dict, plot_ims, to_torch
from head_detection.utils import get_state_dict, plot_ims, to_torch, deadlockcheck
# from head_detection.vision.utils import init_distrexitibuted_mode

try:
    from imageio import imread, imsave
except ImportError:
    # from scipy.misc.pilutil import imread
    from imageio import imread

# -------------- #
import pdb
import datetime as dt
import tracemalloc
import logging
import psutil
# -------------- #

"""
python3 test.py --mode 1

(640, 360) - nHD : 약 초당 12장
(720, 480)

mode1 - 10프레임당 하나씩 하는 경우,
360 프레임 : 12초짜리 영상의 요구시간 14.8초
16.7초 동안 410프레임 처리.`

mode 2 - 모든 프레임에 대해 진행하는 경우,
32프레임:1초에 대해 10.3초 요구.
17초동안 52프레임 처리

바꿀 수 있는 값
utils/__init__py. : headnum과 거리계산(2* boxlen_assembler[startnode]) - 언제든 편의로 바꿀 수 있는 값.
utils/__init__py. : tsuzukuT : 박스가 몇 초 이상 정체되어야 데드락으로 판단할 것인가? 또한 10: 우선 영상은 몇초 이상이어야 하는가?

1) 메모리 확인(누수 여부)
2) 스트레스 테스트 - 무한히 실행
3) 로그 체크

실시간으로 rtsp 실행 시, error while decoding mb bytestream 에러가 발생.
해결책은 캡쳐기능을 다른 스레드에 배치하고, 캡쳐이미지를 활용하는 함수를 스레드에 배치하는 것.
이를 통해, rtsp가 스트리밍되는 동안 추가 처리에서 캡처된 프레임을 사용하고 파이프라인에 지연을 생성할 때 발생하는 이런 오류 방지가 가능.
"""
parser = argparse.ArgumentParser(description='Testing script')
parser.add_argument('--test_dataset', default = '/workspace/HeadHunter/gangreungcam/img1', help='Dataset .txt file')
parser.add_argument('--pretrained_model', default = '/workspace/HeadHunter/weights/FT_R50_epoch_24.pth', help='resume net for retraining')
parser.add_argument('--plot_folder', default = '/workspace/HeadHunter/save_dir/', help='Location to plot results on images')

parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

parser.add_argument('--benchmark', default='Combined', help='Benchmark for training/validation')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--min_size', default=800, type=int, help='Optionally plot first N images of test')
parser.add_argument('--max_size', default=1400, type=int, help='Optionally plot first N images of test')

parser.add_argument('--ext', default='.jpg', type=str, help='Image file extensions')
parser.add_argument('--outfile', help='Location to save results in mot format')

parser.add_argument('--backbone', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--context', default = 'cpm', help='Whether to use context model')
parser.add_argument('--use_deform', default=False, type=bool, help='Use Deformable SSH')
parser.add_argument('--det_thresh', default=0.3, type=float, help='Number of workers used in dataloading')
parser.add_argument('--default_filter', default=False, type=bool, help='Use old filters')
parser.add_argument('--soft_nms', default=False, type=bool, help='Use soft nms?')
parser.add_argument('--upscale_rpn', default=False, type=bool, help='Upscale RPN feature maps')

###
parser.add_argument('--mode', default = 1, type = int, help = 'if 1 : img per frame, if 2 : img per frame(needs more time) and 3: video in /usevideo')
parser.add_argument('--scale', default = 'origin', help = 'result video pixel, width_height')
###

args = parser.parse_args()

##################################
## Set device and config ##########
##################################
if torch.cuda.is_available():
    device = torch.device('cuda')
cfg = None
if args.backbone == "mobile0.25":
    cfg = cfg_mnet
elif args.backbone == "resnet50":
    cfg = cfg_res50_4fpn
elif args.backbone == "resnet152":
    cfg = cfg_res152
else:
    raise ValueError("Invalid configuration")

    
##########################
# 실시간 비디오 #
# 초당 3장을 하니까, 0.3초당 1번 필요. 30프레임에서는 10프레임딩 1회, 60프레임에서는 20프레임당 1회 연산 필요#
##########################
frame_width, frame_height = 0, 0
cap = cv2.VideoCapture('rtsp://root:root@163.239.25.71:554/cam0_0')

if cap.isOpened == False :
    print("Unable to read Camera!")

fps = cap.get(cv2.CAP_PROP_FPS)
gofps = 0
if fps >= 30 :
    gofps = fps//3
else:
    gofps = fps

videodir = '/workspace/HeadHunter/usevideo/'
videofiles = os.listdir(videodir)
tracemalloc.start(5)


"""
Batch_size를 쓰는 구간
yield : 해당 프로세스의 context를 담은 채로(진행도를 알고 있는 채로) 값을 반환.
for문과 yield 함께 씀으로써, for문 한 번(첫번째 idx) 하고 이를 값 반환, 다음에 fetch_images를 실행하면 (두번째 idx) 실행 후 이를 값 반환... 이 가능.

EX. batch_size가 10이면,
batch_ims = [all_ims[k:k+10] for k in range(0, len(all_ims), 10)]
[[0~9], [10~19], [20~29]...]로 batch로 나뉘게 됨
extend - append와 달리, iterable의 각 항목을 넣음.

get_test_dict - 가령 12번째 배치의 4번째 사진인 경우, 12*10+4+1=45를 넣음.
target_ar은, 이를 통해 기본적인 json을 획득
img_ar은 해당 batch(10개로 이뤄진 배열들)의 img로 이뤄짐.
"""
def fetch_images(idx):
    all_ext = '*'+args.ext   # ext= '.jpg'
    all_ims = sorted(glob(osp.join("./capturevideo/" + videofiles[idx][:-4], all_ext)))
    print('hello, ' + "./capturevideo/"+videofiles[idx][:-4] + '!')
    # all_ims = sorted(glob(osp.join(args.test_dataset, all_ext)))
    batched_ims = [all_ims[k:k+args.batch_size] for k in range(0, len(all_ims), args.batch_size)]
    # print('batch ', batched_ims)
    for b_ind, batch in enumerate(batched_ims):
        img_ar = []
        target_ar = []
        for idx, im in enumerate(batch):
            img_ar.extend(to_torch(imread(im)))
            target_ar.append(get_test_dict((args.batch_size*b_ind)+idx+1))
        yield img_ar, target_ar


def fetch_images_realtime(imgidx, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    # im = im.resize((720, 540))
    imgname = str(imgidx).zfill(10)  # 이번 frame의 이름
    if not os.path.isdir('./tempimg'):
        os.mkdir('./tempimg')

    im.save(f'./tempimg/{imgname}.jpg')

    target_ar = []
    target_ar.append(get_test_dict(imgidx))
    img_ar = []
    img_ar.extend(to_torch(imread(f'./tempimg/{imgname}.jpg')))
    yield img_ar, target_ar

def get_test_dict(idx):
    """
    Get FRCNN style dict
    """

    num_objs = 0
    boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

    return {'boxes': boxes,
            'labels': torch.ones((num_objs,), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
            'visibilities': torch.zeros((num_objs), dtype=torch.float32)}

def create_model(combined_cfg):
    kwargs = {}
    kwargs['min_size'] = args.min_size
    kwargs['max_size'] = args.max_size
    kwargs['box_score_thresh'] = args.det_thresh
    kwargs['box_nms_thresh'] = 0.5
    kwargs['box_detections_per_img'] = 300 # increase max det to max val in our benchmark

    """
    customRCNN({**cfg, **combined_anchors}, False, cpm, False, False, False, **kwargs).cuda()
    이 때 kwargs 역시, 밑의 test처럼 각각의 min_size, max_size 등을 하나하나씩 떼서 언패킹 형태로 보내는 형태임.
    """
    model = customRCNN(cfg=combined_cfg, use_deform=args.use_deform,
                       context=args.context, default_filter=args.default_filter,
                       soft_nms=args.soft_nms, upscale_rpn=args.upscale_rpn,
                       **kwargs).cuda()
    return model

def write_results_files(results):
        files = {}
        for image_id, res in results.items():
            # check if out in keys and create empty list if not
            if args.outfile not in files.keys():
                files[args.outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[args.outfile].append(
                    [image_id, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

@torch.no_grad()
def test(idx):
    # print("Testing FPN. On single GPU without Parallelism")
    cpu_device = torch.device("cpu")

    # Set benchmark related parameters
    """
    **는, 언패킹해서 넣는 효과가 있다.
    즉, cfg는 {'name': 'Resnet50', 'clip': False, 'gpu_train': True, 'batch_size': 4, 'ngpu': 1, 'epoch': 100, 'decay1': 70, 'decay2': 90, 'image_size': 840, 'pretrain': True, 
    'return_layers': {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}, 'in_channel': 256, 'out_channel': 256} 로 일종의 딕셔너리 이고
    combined_anchors 역시 {'anchor_sizes': ((12,), (32,), (64,), (112,), (196,), (256,), (384,), (512,)), 
    'aspect_ratios': ((0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5))} 의 딕셔너리임.
    
    이를 **를 통해 언패킹하여 combined_cfg = {**cfg, **combined_anchors}를 진행하면
    {'name' : 'Resnet50', ..... 'out_channel':256, 'aspect_ratios': .....} 로 각각의 요소를 하나씩 떼서 갖다붙이는 딕셔너리가 됨.
    """
    if args.benchmark == 'ScutHead':
        combined_cfg = {**cfg, **sh_anchors}
    elif args.benchmark == 'CHuman':
        combined_cfg = {**cfg, **ch_anchors}
    elif args.benchmark == 'Combined':
        combined_cfg = {**cfg, **combined_anchors}
    else:
        raise ValueError("New dataset has to be registered")

    # 앵커박스(AnchorBoxGenerator), 백본(Backbone_with_fpn) 등을 담은 모델(그 중 FasterRCNN - torchvision.models.detection.generalized_rcnn)을 반환받음.
    model = create_model(combined_cfg)

    new_state_dict = get_state_dict(model, args.pretrained_model)
    model.load_state_dict(new_state_dict, strict=True)
    model = model.eval()
    results = {}

    """
    fetch를 통해, (imgaes, targets)를 반환받음. 이는 (img_ar, target_ar)임.
    img_ar : args.batch_size개로 되어있는 배열 / targets: 해당하는 이미지의 각 정보를 담을 것(박스, 레이블 img id등)
    """
    imgidx = 0   # img의 idx. 값이 지나치게 커질 때마다 0으로 바꿀 것.

    starttime = time.time()
    # ------------------------------ 30프레임이면 10프레임당, 60프레임이면 20프레임당? 머리 찾기 --------------------------------------#
    # """
    if args.mode == 1:
        try:
            while True:
                ret, frame = cap.read()

                if frame is None:
                    break

                if not ret:
                    thermal_capture.release()
                    thermal_capture = cv2.VideoCapture('rtsp://root:root@163.239.25.71:554/cam0_0')
                    print('Found error; rebuilding stream')

                # frame = cv2.resize(frame, (640, 360))

                # 해당 프레임이 gofps에 나눠떨어질 때마다 headhunter을 실행
                # 해당 프레임을 일단 jpg형식으로 변경 필요
                if (imgidx % (int)(gofps) == 0):
                    for img_ind, (images, targets) in enumerate(fetch_images_realtime(imgidx, frame)) :
                        np_images = [(ims.cpu().numpy() * 255.).astype(np.uint8) for ims in images]  # 넘파이화 된 이미지
                        torch.cuda.synchronize()
                        model_time = time.time()
                        outputs = model(images)  # outputs : customRCNN에 images 넣은 결과. 즉 하나의 배치를 통째로 넣음.
                        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                        for b_ind, (np_im, out, tar) in enumerate(zip(np_images, outputs, targets)):
                            out_dict = {'boxes': out['boxes'].cpu(), 'scores': out['scores'].cpu()}
                            results[tar['image_id'].item()] = out_dict
                            plot_images = plot_ims(np_im, out['boxes'].cpu().numpy())
                            # imsave(osp.join(args.plot_folder, str(imgidx).zfill(10) + '.jpg'),
                            #        plot_images)  # (batch_size * img의 idx)의 이름으로 저장하는 거임 pass
                # 나눠떨어지지 않으면, 그냥 이미지를 고대로 저장
                else :
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # imsave(osp.join(args.plot_folder, str(imgidx).zfill(10) + '.jpg'), frame)

                # 프레임 당 1씩 증가시킴. 무한히 커지는 것 방지하기 위해, 일정 수가 되면 0으로 초기화
                # 이를 위해, 이미지 이름이 중복되는 걸 방지하기 위해서라도 오래 된 이미지는 삭제하는 과정 역시 필요
                imgidx += 1
                if imgidx >= 30 * (2**26) :
                    imgidx = 0
                if cv2.waitKey(1) > 0 :
                    break
                # cv2.imshow('frame', frame)
        except KeyboardInterrupt:
            endtime = time.time()
            print('--------------------------------Mode 1 Elapsed Time = ', endtime - starttime, '---------------------------------------')
            print('--------------------------------# of Frame = ', imgidx, ' seconds = ', imgidx // fps, ' --------------------------')

    # -------------------------- 1프레임당 머리 찾기 -------------------------------- #
    elif args.mode == 2:
        try:
            flag = 1
            model_process_time = 0
            npimg_time = 0
            end_time = 0

            while True:
                ret, frame = cap.read()

                if frame is None:
                    break

                if not ret:
                    thermal_capture.release()
                    thermal_capture = cv2.VideoCapture('rtsp://root:root@163.239.25.71:554/cam0_0')
                    print('Found error; rebuilding stream')

                frame = cv2.resize(frame, (640, 360))

                # 무조건 1프레임당 하나씩 머리를 찾음.
                for img_ind, (images, targets) in enumerate(fetch_images_realtime(imgidx, frame)):
                    np_images = [(ims.cpu().numpy() * 255.).astype(np.uint8) for ims in images]  # 넘파이화 된 이미지

                    torch.cuda.synchronize()

                    model_time = time.time()
                    outputs = model(images)  # outputs : customRCNN에 images 넣은 결과. 즉 하나의 배치를 통째로 넣음.
                    model_endtime = time.time()
                    model_process_time = model_endtime - model_time
                    print('model proecess time ', model_process_time)

                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                    for b_ind, (np_im, out, tar) in enumerate(zip(np_images, outputs, targets)):
                        out_dict = {'boxes': out['boxes'].cpu(), 'scores': out['scores'].cpu()}
                        results[tar['image_id'].item()] = out_dict
                        plot_images = plot_ims(np_im, out['boxes'].cpu().numpy())
                        imsave(osp.join(args.plot_folder, str(imgidx).zfill(3) + '.jpg'),
                                plot_images)  # (batch_size * img의 idx)의 이름으로 저장하는 거임 pass

                imgidx += 1
                
                # imgidx가 일정치 이상이면, 0으로 초기화
                if imgidx >= 100:
                    imgidx = 0
                if flag == 1:
                    endtime = time.time()

                flag += 1

                # print(psutil.virtual_memory())
                
                # 나눠떨어지지 않으면, 그냥 이미지를 고대로 저장
        except KeyboardInterrupt:
            finalendtime = time.time()
            print('-------------------------------- Elapsed time = ', finalendtime - starttime, '---------------------------------------')
            print('-------------------------------- Mode 2 processing time = ', endtime - starttime, '---------------------------------------')
            print('--------------------------------# of Frame = ', imgidx, ' seconds = ', imgidx // fps, ' --------------------------')
            print('--------------------------------model processing time = ', model_process_time, ' --------------------------')
    # """


        # -----------------------------------------mode가 3: 주어진 동영상으로 실시----------------------------------------------------------------------------#
    elif args.mode == 3 :
        flag = 1
        crowdbox_assembler = []
        model_process_time = 0
        for img_ind, (images, targets) in tqdm(enumerate(fetch_images(idx))):
            np_images = [(ims.cpu().numpy()*255.).astype(np.uint8) for ims in images]  #넘파이화 된 이미지
            torch.cuda.synchronize()

            model_time = time.time()
            outputs = model(images)   # outputs : customRCNN에 images 넣은 결과. 즉 하나의 배치를 통째로 넣음.

            end_time = time.time()
            if flag == 1 :
                print('--------------------------Elapsed Time = ', end_time - model_time)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            for b_ind, (np_im, out, tar) in enumerate(zip(np_images, outputs, targets)):
                out_dict = {'boxes': out['boxes'].cpu(), 'scores': out['scores'].cpu()}
                results[tar['image_id'].item()] = out_dict
                plot_images = plot_ims(np_im, out['boxes'].cpu().numpy())
                # plot_images, curimgcrowd = plot_ims(np_im, out['boxes'].cpu().numpy())
                """
                # ------------------첫번째 이미지는 그냥 crowdbox_assembler에 삽입----------------------------- #
                # thisimagescrowd = 방금 확인한 이미지의 군중의 좌표 [[(좌표),(좌표)], [(),()]]형태
                # crowdbox_assembler = [[(좌표), (좌표), 인식횟수], [(), (), #]] 형태. 이 때 #는 중복군중인식 횟수
                if img_ind == 0:
                    for i in range(len(curimgcrowd)):
                        curimgcrowd[i].append(1)
                        crowdbox_assembler.append(curimgcrowd[i])
                        # print('now crowd ', crowdbox_assembler)
                else :
                    is_deadlock, where_deadlock = deadlockcheck(crowdbox_assembler, curimgcrowd)
                    if is_deadlock == True :
                        cv2.rectangle(plot_images, (where_deadlock[0][0], where_deadlock[0][1]), (where_deadlock[1][0], where_deadlock[1][1]),
                                      (255, 0, 0), 2)
                # ------------------이후에 다음 이미지가 오면 이전과 IoU를 비교해가며 확인------------------------- #
                """

                imsave(osp.join(args.plot_folder, str((args.batch_size*img_ind)+b_ind+1).zfill(10) + '.jpg'), plot_images)   # (batch_size * img의 idx)의 이름으로 저장하는 거임 pass
            # flag += 1
        # ----------------------------------------------------------------------------------------------------------------------#
        # n초 이상(tsuzukuT) 지속되면, 이는 곧 정체로 확인.
        # is_deadlock = deadlockcheck(crowdbox_assembler)
        # print(crowdbox_assembler)
        # ----------------------------------------------------------------------------------------------------------------------#


    write_results_files(results)

def imageparse(idx):
    if not videofiles[idx].endswith('mp4'):
        return

    print(videodir + videofiles[idx])
    vidcap = cv2.VideoCapture(videodir + videofiles[idx])
    h, w = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT), vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(h, w)

    # 원본화질로 뽑고싶은 경우
    # resize(너비, 높이)
    if args.scale == 'origin':
        print('make video with origin scale!')
        newsize = (int(w), int(h))
    else:  # 아니면, ,기준으로 왼쪽은 너비, 오른쪽은 높이
        newh, neww = args.scale.split('_')[0], args.scale.split('_')[1]
        newsize = (int(neww), int(newh))

    if h > w:  # 높이가 더 큰 영상인 경우(세로로 찍은 경우), 사이즈 바꾸고 180도 회전시켜야 됨.
        newsize = (newsize[1], newsize[0])

    count = 0

    if not os.path.exists("./capturevideo/" + videofiles[idx][:-4]):
        os.makedirs("./capturevideo/" + videofiles[idx][:-4])
        print('make dir - ' + "./capturevideo/" + videofiles[idx][:-4])
    else:
        print('already there is dir! ' + "./capturevideo/" + videofiles[idx][:-4])

    while (vidcap.isOpened()):
        ret, image = vidcap.read()
        # 이미지 사이즈 960x540으로 변경
        if image is None:
            break

        image = cv2.resize(image, newsize)

        if h > w:
            image = cv2.rotate(image, cv2.ROTATE_180)

        # 프레임당 하나씩 이미지 추출
        if (int(vidcap.get(1)) % 1 == 0):
            if count % 900 == 0:  # 30초마다 출력
                print('30sec over', end=' ')
                # print('Saved frame number : ' + str(int(vidcap.get(1))))
            # 추출된 이미지가 저장되는 경로
            cv2.imwrite("./capturevideo/" + videofiles[idx][:-4] + '/' + str(count).zfill(10) + ".jpg", image)
            # print(os.listdir("./capturevideo/" + videofiles[idx][:-4]))
            # print('Saved frame%d.jpg' % count)
        count += 1

    vidcap.release()
    print(len(os.listdir("./capturevideo/" + videofiles[idx][:-4])))


def videomaker(idx):
    if not videofiles[idx].endswith('mp4'):
        return

    filepath = args.plot_folder
    # filepath = '/workspace/HeadHunter/save_dir/HT21-31/img'
    useimg = sorted(os.listdir(filepath))
    if not os.path.exists('/workspace/HeadHunter/resultvideo'):
        os.makedirs('/workspace/HeadHunter/resultvideo')
    pathout = '/workspace/HeadHunter/resultvideo/' + videofiles[idx][:-4] + '_' + args.scale + '.mp4'
    print('made video ' + '/workspace/HeadHunter/resultvideo/' + videofiles[idx])
    # print(len(useimg))

    fps = 30
    framearray = []
    print(filepath + useimg[0])
    imgsample = useimg[0]
    h, w, c = cv2.imread(filepath + '/' + imgsample).shape
    print(h, w, c)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    writers = cv2.VideoWriter(pathout, fourcc, fps, (w, h))

    frame = 0
    for idx in tqdm(useimg, desc='processing img....'):
        readed_img = os.path.join(filepath, idx)
        if readed_img.startswith('results.txt'):
            continue

        # print(readed_img)
        img = cv2.imread(readed_img)
        if img is None:  # 강제 Ctrl+C로 인해, None처리되는 경우가 있음.
            continue

        writers.write(img)

    writers.release()
    return


if __name__ == '__main__':
    for i in range(len(videofiles)):
    # for i in range(1):
        ##########################
        # outfile and plot file ##
        ##########################
        if args.plot_folder is not None:
            os.makedirs('/workspace/HeadHunter/save_dir/' + videofiles[i][:-4] + '/img', exist_ok=True)
            args.plot_folder = '/workspace/HeadHunter/save_dir/' + videofiles[i][:-4] + '/img'
        else:
            raise AssertionError("Must provide save directory")
        if args.outfile is None:
            args.outfile = osp.join(args.plot_folder, 'results.txt')

        if videofiles[i] == 'cam9.mp4' or videofiles[i] == 'crowdsample5.mp4' or videofiles[i] == 'crowdsample6.mp4' \
                or videofiles[i].startswith('indiaRush') or videofiles[i].startswith('bandicam'):
            continue
        print('processing ' + videofiles[i] +'....')

        if args.mode == 3 :
            imageparse(i)
        test(i)

        if args.mode == 3:
            videomaker(i)