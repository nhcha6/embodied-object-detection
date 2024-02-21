#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import math
import os
from os.path import join
import numpy as np
import copy
from functools import partial

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone import FPN
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers.batch_norm import get_norm, FrozenBatchNorm2d
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.meta_arch.semantic_seg import SemSegFPNHead
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.layers.mask_ops import paste_masks_in_image
from detectron2.utils.visualizer import Visualizer

from timm import create_model
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from timm.models.resnet import ResNet, Bottleneck
from timm.models.resnet import default_cfgs as default_cfgs_resnet
from timm.models.convnext import ConvNeXt, default_cfgs, checkpoint_filter_fn
import random

from torch.cuda.amp import autocast
import time
import h5py
import json
from detectron2.layers import Conv2d, get_norm
import cv2

@register_model
def convnext_tiny_21k(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    cfg = default_cfgs['convnext_tiny']
    cfg['url'] = 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth'
    model = build_model_with_cfg(
        ConvNeXt, 'convnext_tiny', pretrained,
        default_cfg=cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **model_args)
    return model
    
class CustomRecurrentFPN(FPN):
    def __init__(self, **kwargs):
        self.norm = kwargs['norm']
        self.feat_fusion = kwargs.pop('fusion')
        self.merge_type = kwargs.pop('merge_type')
        self.memory_type = kwargs.pop('memory_type')
        self.memory_feature_weight = kwargs.pop('memory_feature_weight')
        self.map_feature_weight = kwargs.pop('map_feature_weight')
        super().__init__(**kwargs)

        # dimension of extracted image features
        ego_feat_dim = 256
        # dimension of memory features
        mem_feat_dim = 512
        self.memory_dim = mem_feat_dim

        # intialise memory
        self.backbone_type = "recurrent"

        # layers for updating the memory features
        if self.memory_type in ['implicit_memory', 'explicit_map']:
            merge_norm = get_norm(self.norm, ego_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_projection1 = Conv2d(
                512, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection2 = Conv2d(
                512, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection3 = Conv2d(
                512, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )

            self.merge_map_projections = [self.map_merge_projection1, self.map_merge_projection2, self.map_merge_projection3]

    
    def forward(self, x, map_memory, proj_indices, observations, sequence_name=None):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """

        # intialise empty egocentric memory
        # if we want to merge features at the start of the backbone, we need to calculate egocentric memory here before running self.bottom_up
        egocentric_memory = []

        ########################### INITIAL BACKBONE ##################################

        # resnet returns the bottom up features
        # FPN progressively upsamples the features and merges them
        bottom_up_features = self.bottom_up(x, egocentric_memory)
        # print the shape of the bottom up features
        # for k, v in bottom_up_features.items():
        #     print(k, v.shape)

        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        
        ########################## MERGE IN THE MEMORY FEATURES ################################
    
        # use cnn projections to include memory features
        if self.memory_type in ['implicit_memory', 'explicit_map']:
            
            print('enhancing features with map based memory')
            start = time.time()
            egocentric_memory = []
            for i in range(len(map_memory)):
                # project memory features into the image frame
                ego_mem_i = map_memory[i][proj_indices[i]].permute(2, 0, 1).unsqueeze(0).cuda()
                ego_obs_i = observations[i][proj_indices[i]]

                # downsample the memory features to the size of the highest resolution feature map 
                # ego_mem_i = F.interpolate(ego_mem_i.to(torch.float32), scale_factor=0.125, mode="bilinear")
                ego_mem_i = F.avg_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
                # ego_mem_i = F.max_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
                ego_mem_i = ego_mem_i.squeeze(0)

                # add to the list of memory features
                egocentric_memory.append(ego_mem_i)

            egocentric_memory = torch.stack(egocentric_memory)

            # print('using cnn to merge precomputed features')
            new_results = []
            for i, res in enumerate(results):
                b, d, w, h = res.shape
                res_dtype = res.dtype

                # downsample the memory features to be used next iterations
                egocentric_memory = F.avg_pool2d(egocentric_memory.to(torch.float32), kernel_size=2, stride=2).to(torch.half)

                # print min and max of res and mem
                # print('res mean: ', torch.mean(res.abs()))

                # apply the 1x1 convolution to the memory features to map to same space as the image features
                mem = self.merge_map_projections[i](egocentric_memory.to(torch.float32).cuda())
                
                # scale the mem
                mem = mem*self.map_feature_weight
                # print(scale_factor)

                # merge by summing the two feature spaces
                if self.feat_fusion == 'sum':
                    new_res = mem+res
                elif self.feat_fusion == 'mem_only':
                    new_res = mem
                elif self.feat_fusion == 'image_only':
                    new_res = res

                # add to new results
                new_results.append(new_res.to(res_dtype))
            
            # update the results
            results = new_results

            # start = time.time()

        else:
            # print('continuing with image features only')
            pass

        ################################# TOP BLOCK OF FEATURE BACKBONE #####################################
        
        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)

        # # print the shape of the results
        # for f, res in zip(self._out_features, results):
        #     print(f, res.shape)

        # return the updated features
        return {f: res for f, res in zip(self._out_features, results)}, map_memory[0]

class CustomResNet(ResNet):
    def __init__(self, **kwargs):
        self.out_indices = kwargs.pop('out_indices')
        super().__init__(**kwargs)


    def forward(self, x):
        # print('\n')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        # print(x.shape)
        ret = [x]
        x = self.layer1(x)
        # print(x.shape)
        ret.append(x)
        x = self.layer2(x)
        # print(x.shape)
        ret.append(x)
        x = self.layer3(x)
        # print(x.shape)
        ret.append(x)
        x = self.layer4(x)
        # print(x.shape)
        ret.append(x)

        # for i in self.out_indices:
        #     print(ret[i].shape)
        return [ret[i] for i in self.out_indices]


    def load_pretrained(self, cached_file):
        data = torch.load(cached_file, map_location='cpu')
        if 'state_dict' in data:
            self.load_state_dict(data['state_dict'])
        else:
            self.load_state_dict(data)


class CustomResNetMap(ResNet):
    def __init__(self, **kwargs):
        self.out_indices = kwargs.pop('out_indices')
        super().__init__(**kwargs)

    def init_merge_layers(self, merge_type, norm):
        self.merge_type = merge_type
        self.norm = norm

        # dimension of extracted image features
        ego_feat_dim = 256
        # dimension of memory features
        mem_feat_dim = 256

        if self.merge_type == "cnn_precomputed_bottom":
            merge_norm = get_norm(self.norm, ego_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_projection = Conv2d(
                mem_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )

    def forward(self, x, memory):
        # print('\n')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        ret = [x]
        x = self.layer1(x)
        ret.append(x)
        x = self.layer2(x)
        # print(x.shape)
        ret.append(x)
        x = self.layer3(x)
        # print(x.shape)
        ret.append(x)
        x = self.layer4(x)
        # print(x.shape)
        ret.append(x)

        # for i in self.out_indices:
        #     print(ret[i].shape)
        return [ret[i] for i in self.out_indices]


    def load_pretrained(self, cached_file):
        data = torch.load(cached_file, map_location='cpu')
        if 'state_dict' in data:
            self.load_state_dict(data['state_dict'])
        else:
            self.load_state_dict(data)

model_params = {
    'resnet50_in21k': dict(block=Bottleneck, layers=[3, 4, 6, 3]),
    'resnet50_in21k_map': dict(block=Bottleneck, layers=[3, 4, 6, 3])
}


def create_timm_resnet(variant, out_indices, pretrained=False, **kwargs):
    params = model_params[variant]
    default_cfgs_resnet['resnet50_in21k'] = \
        copy.deepcopy(default_cfgs_resnet['resnet50'])
    default_cfgs_resnet['resnet50_in21k']['url'] = \
        'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'
    default_cfgs_resnet['resnet50_in21k']['num_classes'] = 11221

    return build_model_with_cfg(
        CustomResNet, variant, pretrained,
        default_cfg=default_cfgs_resnet[variant],
        out_indices=out_indices,
        pretrained_custom_load=True,
        **params,
        **kwargs)

def create_timm_resnet_map(variant, out_indices, pretrained=False, **kwargs):
    params = model_params[variant]
    default_cfgs_resnet[variant] = \
        copy.deepcopy(default_cfgs_resnet['resnet50'])
    default_cfgs_resnet[variant]['url'] = \
        'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'
    default_cfgs_resnet[variant]['num_classes'] = 11221

    return build_model_with_cfg(
        CustomResNetMap, variant, pretrained,
        default_cfg=default_cfgs_resnet[variant],
        out_indices=out_indices,
        pretrained_custom_load=True,
        **params,
        **kwargs)

class LastLevelP6P7_P5(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "p5"
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        # print(p6.shape)
        # print(p7.shape)
        return [p6, p7]


def freeze_module(x):
    """
    """
    for p in x.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(x)
    return x


class TIMM(Backbone):
    def __init__(self, base_name, out_levels, freeze_at=0, norm='FrozenBN', pretrained=False):
        super().__init__()
        out_indices = [x - 1 for x in out_levels]
        if base_name in model_params:
            if base_name == 'resnet50_in21k':
                self.base = create_timm_resnet(
                    base_name, out_indices=out_indices, 
                    pretrained=False)
            else:
                self.base = create_timm_resnet_map(
                    base_name, out_indices=out_indices, 
                    pretrained=False)
        elif 'eff' in base_name or 'resnet' in base_name or 'regnet' in base_name:
            self.base = create_model(
                base_name, features_only=True, 
                out_indices=out_indices, pretrained=pretrained)
        elif 'convnext' in base_name:
            drop_path_rate = 0.2 \
                if ('tiny' in base_name or 'small' in base_name) else 0.3
            self.base = create_model(
                base_name, features_only=True, 
                out_indices=out_indices, pretrained=pretrained,
                drop_path_rate=drop_path_rate)
        else:
            assert 0, base_name
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction']) \
            for i, f in enumerate(self.base.feature_info)] 
        self._out_features = ['layer{}'.format(x) for x in out_levels]
        self._out_feature_channels = {
            'layer{}'.format(l): feature_info[l - 1]['num_chs'] for l in out_levels}
        self._out_feature_strides = {
            'layer{}'.format(l): feature_info[l - 1]['reduction'] for l in out_levels}
        self._size_divisibility = max(self._out_feature_strides.values())
        if 'resnet' in base_name:
            self.freeze(freeze_at)
        if norm == 'FrozenBN':
            self = FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def freeze(self, freeze_at=0):
        """
        """
        if freeze_at >= 1:
            print('Frezing', self.base.conv1)
            self.base.conv1 = freeze_module(self.base.conv1)
        if freeze_at >= 2:
            print('Frezing', self.base.layer1)
            self.base.layer1 = freeze_module(self.base.layer1)

    def forward(self, x):
        features = self.base(x)
        ret = {k: v for k, v in zip(self._out_features, features)}
        return ret
    
    @property
    def size_divisibility(self):
        return self._size_divisibility
    
class MapTIMM(TIMM):
    def __init__(self, base_name, out_levels, freeze_at=0, norm='FrozenBN', pretrained=False, merge_type='image_features_only'):
        super().__init__(base_name, out_levels, freeze_at=freeze_at, norm=norm, pretrained=pretrained)
        self.base.init_merge_layers(merge_type, norm)

    def forward(self, x, memory):
        features = self.base(x, memory)
        ret = {k: v for k, v in zip(self._out_features, features)}
        return ret

@BACKBONE_REGISTRY.register()
def build_timm_backbone(cfg, input_shape):
    model = TIMM(
        cfg.MODEL.TIMM.BASE_NAME, 
        cfg.MODEL.TIMM.OUT_LEVELS,
        freeze_at=cfg.MODEL.TIMM.FREEZE_AT,
        norm=cfg.MODEL.TIMM.NORM,
        pretrained=cfg.MODEL.TIMM.PRETRAINED,
    )
    return model

@BACKBONE_REGISTRY.register()
def build_timm_backbone_map(cfg, input_shape):
    model = MapTIMM(
        cfg.MODEL.TIMM.BASE_NAME, 
        cfg.MODEL.TIMM.OUT_LEVELS,
        freeze_at=cfg.MODEL.TIMM.FREEZE_AT,
        norm=cfg.MODEL.TIMM.NORM,
        pretrained=cfg.MODEL.TIMM.PRETRAINED,
        merge_type=cfg.MODEL.MAP_MERGE_TYPE,
    )
    return model

@BACKBONE_REGISTRY.register()
def build_p67_timm_fpn_backbone(cfg, input_shape):
    """
    """
    bottom_up = build_timm_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p67_timm_fpn_backbone_map_conditioned(cfg, input_shape):
    """
    """
    bottom_up = build_timm_backbone_map(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    merge_type = cfg.MODEL.MAP_MERGE_TYPE
    feat_fusion = cfg.MODEL.MAP_FEAT_FUSION 
    backbone = CustomMapFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        merge_type=merge_type,
        fusion = feat_fusion,
        memory_feature_weight = cfg.MODEL.MEMORY_FEATURE_WEIGHT,
        map_feature_weight = cfg.MODEL.MAP_FEATURE_WEIGHT,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p67_timm_fpn_backbone_recurrent(cfg, input_shape):
    """
    """
    bottom_up = build_timm_backbone_map(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    merge_type = cfg.MODEL.MAP_MERGE_TYPE
    feat_fusion = cfg.MODEL.MAP_FEAT_FUSION
    memory_type = cfg.MODEL.MEMORY_TYPE

    backbone = CustomRecurrentFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        merge_type=merge_type,
        fusion = feat_fusion,
        memory_type = memory_type,
        memory_feature_weight = cfg.MODEL.MEMORY_FEATURE_WEIGHT,
        map_feature_weight = cfg.MODEL.MAP_FEATURE_WEIGHT,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p67_timm_fpn_backbone_mamba(cfg, input_shape):
    """
    """
    bottom_up = build_timm_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    merge_type = cfg.MODEL.MAP_MERGE_TYPE
    feat_fusion = cfg.MODEL.MAP_FEAT_FUSION 
    backbone = CustomMambaFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        merge_type=merge_type,
        fusion = feat_fusion,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p35_timm_fpn_backbone(cfg, input_shape):
    """
    """
    bottom_up = build_timm_backbone(cfg, input_shape)
    
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone