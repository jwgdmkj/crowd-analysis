#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict

import torch
import torchvision.models as models
from torchvision.models.detection.rpn import AnchorGenerator

from head_detection.models.fast_rcnn import FasterRCNN
from head_detection.models.net import BackBoneWithFPN
from head_detection.models.net import MobileNetV1 as MobileNetV1

import tracemalloc
"""
Backbone 생성

"""
def create_backbone(cfg, use_deform=False,
                    context=None, default_filter=False):
    """Creates backbone """
    in_channels = cfg['in_channel']    # 256
    if cfg['name'] == 'Resnet50':  # 이걸 진행. ResnotModel50을 pretrain=True 형태로 불러온다.
        feat_ext = models.resnet50(pretrained=cfg['pretrain'])    # feat_ext = models.resnet50(True)
        if len(cfg['return_layers']) == 3:
            in_channels_list = [
                in_channels * 2,
                in_channels * 4,
                in_channels * 8,
            ]
        elif len(cfg['return_layers']) == 4:  # 이걸 진행
            in_channels_list = [
                    in_channels,
                    in_channels * 2,
                    in_channels * 4,
                    in_channels * 8,
            ]
        else:
            raise ValueError("Not yet ready for 5FPN")
    elif cfg['name'] == 'Resnet152':
        feat_ext = models.resnet152(pretrained=cfg['pretrain'])
        in_channels_list = [
            in_channels,
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
        ]
    elif cfg['name'] == 'mobilenet0.25':
        feat_ext = MobileNetV1()
        in_channels_list = [
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
        ]
        if cfg['pretrain']:
            checkpoint = torch.load("./Weights/mobilenetV1X0.25_pretrain.tar",
                                    map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
                # load params
            feat_ext.load_state_dict(new_state_dict)
    else:
        raise ValueError("Unsupported backbone")

    # print(feat_ext)

    out_channels = cfg['out_channel']
    backbone_with_fpn = BackBoneWithFPN(feat_ext, cfg['return_layers'],   # feat_ext = ResNet50, cfg['return_layers']  = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
                                        in_channels_list,   # [256, 512, 1024, 2048]
                                        out_channels,   # 256
                                        context_module=context,
                                        use_deform=use_deform,
                                        default_filter=default_filter)
    return backbone_with_fpn    # __init__ 실행한 후 이를 return


"""
본격젹인 머리 찾는 부분
customRCNN({**cfg, **combined_anchors}, False, cpm, False, False, False, **kwargs).cuda()
"""
def customRCNN(cfg, use_deform=False,
              ohem=False, context=None, custom_sampling=False,
              default_filter=False, soft_nms=False,
              upscale_rpn=False, median_anchors=True,
              **kwargs):
    
    """
    Calls a Faster-RCNN head with custom arguments + our backbone
    backbone_with_fpn을 반환받음.
    """
    backbone_with_fpn = create_backbone(cfg=cfg, use_deform=use_deform,
                                        context=context,
                                        default_filter=default_filter)

    # default = T
    if median_anchors:
        anchor_sizes = cfg['anchor_sizes'] # 'anchor_sizes': ((12,), (32,), (64,), (112,), (196,), (256,), (384,), (512,))
        aspect_ratios = cfg['aspect_ratios']  # 'aspect_ratios': ((0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.5, 1.0, 1.5))

        """
        AnchorGenetor - https://tutorials.pytorch.kr/intermediate/torchvision_tutorial.html
        이에 의하면, [12, 32, 64, ...]의 서로 다른 크기와 (0.5, 1.0, 1.5)의 다른 측면 비율(Aspect ratio)를 가진
        8x3개의 앵커를 공간위치마다 생성하도록 한다고 한다.
        이 aspect_ratio와 anchor_sizes에 관해서는 https://herbwood.tistory.com/10 
        또는 https://github.com/chullhwan-song/Reading-Paper/issues/184
        
        아무튼 만약 input사이즈를 줄인다면, 앵커 사이즈도 줄여야 하지 않을까? 
        각도를 고려할 때 큰 앵커사이즈는 필요 없을 수도 있으니, 큰 앵커사이즈를 없애도 될 지도....
        """
        rpn_anchor_generator = AnchorGenerator(anchor_sizes,
                                           aspect_ratios)
        kwargs['rpn_anchor_generator'] = rpn_anchor_generator

    # default = F
    if custom_sampling:
        # Random hand thresholding ablation experiment to understand difference
        # in behaviour of Body vs head bounding boxes
        kwargs['rpn_fg_iou_thresh'] = 0.5
        kwargs['box_bg_iou_thresh'] = 0.4
        kwargs['box_positive_fraction'] = 0.5
        # kwargs['box_nms_thresh'] = 0.7

    kwargs['cfg'] = cfg
    model = FasterRCNN(backbone_with_fpn, num_classes=2, ohem=ohem, soft_nms=soft_nms,
                       upscale_rpn=upscale_rpn, **kwargs)
    return model
