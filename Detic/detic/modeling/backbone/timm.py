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

# define the colour pallette
palette = np.array([
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [0, 255, 255],  # Cyan
    [255, 0, 255],  # Magenta
    [128, 0, 0],    # Maroon
    [0, 128, 0],    # Green (Dark)
    [0, 0, 128],    # Navy
    [128, 128, 0],  # Olive
    [0, 128, 128],  # Teal
    [128, 0, 128],  # Purple
    [192, 192, 192],  # Silver
    [128, 128, 128],  # Gray
    [255, 165, 0],  # Orange
    [128, 0, 0],  # Brown
    [0, 0, 128],  # Navy (Dark)
    [128, 0, 128],  # Purple (Dark)
    [0, 128, 0],  # Green (Dark)
    [0, 128, 128],  # Teal (Dark)
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [0, 255, 255],  # Cyan
    [255, 0, 255],  # Magenta
    [128, 0, 0],    # Maroon
    [0, 128, 0],    # Green (Dark)
    [0, 0, 128],    # Navy
    [128, 128, 0],  # Olive
    [0, 128, 128],  # Teal
    [128, 0, 128],  # Purple
    [192, 192, 192],  # Silver
    [128, 128, 128],  # Gray
    [255, 165, 0],  # Orange
    [128, 0, 0],  # Brown
    [0, 0, 128],  # Navy (Dark)
    [128, 0, 128],  # Purple (Dark)
    [0, 128, 0],  # Green (Dark)
    [0, 128, 128],  # Teal (Dark)
    [0, 0, 0]
])

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

class CustomSemSegFPNHead(SemSegFPNHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features)
        # if self.training:
        #     return None, self.losses(x, targets)
        # else:
        
        x = F.interpolate(
            x, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        return x, {}

    def layers(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        # x = self.predictor(x)
        return x

class CustomMambaFPN(FPN):
    def __init__(self, **kwargs):
        self.merge_type = kwargs.pop('merge_type')
        self.norm = kwargs['norm']
        self.feat_fusion = kwargs.pop('fusion')
        super().__init__(**kwargs)
    
    def forward(self, x):
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
        bottom_up_features = self.bottom_up(x)
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

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

class CustomMapFPN(FPN):
    def __init__(self, **kwargs):
        self.merge_type = kwargs.pop('merge_type')
        self.norm = kwargs['norm']
        self.feat_fusion = kwargs.pop('fusion')
        self.memory_feature_weight = kwargs.pop('memory_feature_weight')
        self.map_feature_weight = kwargs.pop('map_feature_weight')
        super().__init__(**kwargs)

        # dimension of extracted image features
        ego_feat_dim = 256
        # dimension of memory features
        mem_feat_dim = 256

        # memory feautres
        self.backbone_type = 'precomputed_map'

        # additional models defined here to merge sequences
        if self.merge_type == 'gru_precomputed':
            self.map_merge_gru1 = nn.GRUCell(mem_feat_dim, ego_feat_dim, bias=True)
            self.map_merge_gru2 = nn.GRUCell(mem_feat_dim, ego_feat_dim, bias=True)
            self.map_merge_gru3 = nn.GRUCell(mem_feat_dim, ego_feat_dim, bias=True)
            self.merge_rnns = [self.map_merge_gru1, self.map_merge_gru2, self.map_merge_gru3]

            # change default LSTM initialization
            for rnn in self.merge_rnns:
                noise = 0.01
                rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(rnn.weight_hh)
                rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(rnn.weight_ih)
                rnn.bias_hh.data = torch.zeros_like(rnn.bias_hh)  # redundant with bias_ih
                rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(rnn.bias_ih)
        
        elif self.merge_type in ['cnn_precomputed', 'implicit_memory', 'image_projection']:
            merge_norm = get_norm(self.norm, ego_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_projection1 = Conv2d(
                mem_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection2 = Conv2d(
                mem_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection3 = Conv2d(
                mem_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )

            self.merge_projections = [self.map_merge_projection1, self.map_merge_projection2, self.map_merge_projection3]

        elif self.merge_type in ['semantic_gt', 'map_gt']:
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

            self.merge_projections = [self.map_merge_projection1, self.map_merge_projection2, self.map_merge_projection3]

        # layers for updating the memory features
        if self.merge_type == 'implicit_memory':
            merge_norm = get_norm(self.norm, mem_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_memory_projection = Conv2d(
                ego_feat_dim, 256, kernel_size=1, bias=True, norm=merge_norm
            )

            # a 1x1 convolution in 1 dimension to reduce combine memory and image features and project them to a new representation space
            self.map_merge_memory_update = torch.nn.Conv1d(512, 256, kernel_size=1)

            # # an rnn to update the memory features
            # self.map_merge_rnn = nn.GRUCell(256, mem_feat_dim, bias=True)

            # # change default LSTM initialization
            # noise = 0.01
            # self.map_merge_rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.weight_hh)
            # self.map_merge_rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.weight_ih)
            # self.map_merge_rnn.bias_hh.data = torch.zeros_like(self.map_merge_rnn.bias_hh)  # redundant with bias_ih
            # self.map_merge_rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.bias_ih)
            
        # layers for the image_projection test
        if self.merge_type == 'image_projection':
            merge_norm  = get_norm(self.norm, ego_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_forward_projection = Conv2d(
                ego_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )

    
    def validate_projection_indices(self, ego_mem_i, sequence_name):    
        # load the memory features from file
        h5file = h5py.File(os.path.join('../Semantic-MapNet/data/training/', 'smnet_training_data_memory', sequence_name[0:13]) + '.h5', 'r')
        map_memory = np.array(h5file['memory'])
        semmap = np.array(h5file['pred_map'])
        map_memory = np.transpose(map_memory[0], (1, 2, 0))
        h5file.close()

        # load the raw projection information
        h5file = h5py.File(os.path.join('../Semantic-MapNet/data/training/', 'smnet_training_data', sequence_name), 'r')
        projection_indices = np.array(h5file['projection_indices'], dtype=np.float32)
        masks_outliers = np.array(h5file['masks_outliers'], dtype=np.bool)
        sensor_positions = np.array(h5file['sensor_positions'], dtype=np.float32)
        h5file.close()

        # load scene information
        resolution = 0.02 # topdown resolution'
        semmap_info = json.load(open('../Semantic-MapNet/data//semmap_GT_info.json', 'r'))
        map_world_shift = np.array(semmap_info[sequence_name[0:13]]['map_world_shift'])

        # transfer to Pytorch
        projection_indices = torch.FloatTensor(projection_indices)
        masks_outliers = torch.BoolTensor(masks_outliers)
        sensor_positions = torch.FloatTensor(sensor_positions)
        map_world_shift = torch.FloatTensor(map_world_shift)
        projection_indices -= map_world_shift
        pixels_in_map = (projection_indices[:,:,:, [0,2]] / resolution).round().long()

        # get a list of all grid indices that are viewed in this sequence
        pixels_in_map = pixels_in_map.numpy()

        # project ego_memory features back to the image 
        predicted_mem = ego_mem_i.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_mem = map_memory[pixels_in_map[0, :, :, 1], pixels_in_map[0, :, :, 0], :]
        print('Calculated memory features: ', predicted_mem.shape)
        print('Ground truth memory features: ', gt_mem.shape)

        # check it is the same as the original memory features
        # print(predicted_mem[0, 0, :])
        # print(gt_mem[0, 0, :])
        # flatten both arrays
        predicted_mem_flat = predicted_mem.reshape(-1, 256)
        gt_mem_flat = gt_mem.reshape(-1, 256)
        # calculate the distance between each corresponding element in the arrays
        dist = np.linalg.norm(predicted_mem_flat - gt_mem_flat, axis=1)
        # print the mean of the distance
        print('Mean distance between calculated and ground truth memory features: ', np.mean(dist))
        # for comparison, randomly shuffle the memory features and repeat the calculation
        np.random.shuffle(predicted_mem_flat)
        dist = np.linalg.norm(predicted_mem_flat - gt_mem_flat, axis=1)
        print('Mean distance between random features ', np.mean(dist))
        
        # assess distance between features
        # map_w, map_h = map_memory.shape[0], map_memory.shape[1]
        # map_mem_flat = map_memory.reshape(-1, 256)
        # predicted_mem_flat = predicted_mem.reshape(-1, 256)
        # from scipy.spatial.distance import cdist
        # dist = cdist(map_mem_flat, predicted_mem_flat, 'euclidean')
        # closest_indices = np.argmin(dist, axis=1)
        # # convert flattened indices to 2D indices
        # closest_indices = np.unravel_index(closest_indices, (map_w, map_h))
        # closest_indices = closest_indices.reshape(480, 640, 2)
        # print(closest_indices)
        # print(pixels_in_map[0])

    def visualise_clip_image_features(self, image_features, zs_weight, fig_name = 'image_features', mask=None, thresh = 0.4, visualise = True):
        
        ################### USE FEATURES TO GET CLASSIFICATIONS ######################3
        
        # get the shape
        _, c, w, h = image_features.shape
        
        # flatten features    
        flat_features = image_features.squeeze(0).permute(1,2,0).reshape(-1, 512)
        
        # normalise the box features
        norm_feautures = 50.0 * torch.nn.functional.normalize(flat_features, p=2, dim=1)
        
        # dot product with clip embeddings to get logits
        image_feat_scores = torch.mm(norm_feautures, zs_weight)[:,:20]
        
        # apply softmax
        image_feat_scores = image_feat_scores.softmax(dim=1)
        
        # get the maximum score for each pixel
        max_scores, max_indices = torch.max(image_feat_scores, dim=1)

        # reshape the max indices and the scores
        max_indices = max_indices.reshape(w, h).cpu().numpy()
        max_scores = max_scores.reshape(w, h, 1)

        # # convert to rgb for displaying the results
        # max_indices_colour = palette[max_indices]

        ################# APPLY THRESHOLD TO EITHER MASK OR SCORES #################

        # use the input mask to threshold the results 
        if mask is not None:
            # max_scores = max_scores * mask.reshape(w,h,1)
            max_scores = mask.reshape(w,h,1)

        # scale the colour by the score for visualisation
        # max_indices = max_indices * max_scores.cpu().numpy()
        
        # alternatively, generate an explicit map by apply a threshold
        max_indices[max_scores.squeeze(2).cpu().numpy() < thresh] = -1

        # convert to rgb to display the results
        max_indices_colour = palette[max_indices]

        # convert to 32 bit signed int
        max_indices_colour = max_indices_colour.astype(np.uint8)

        ################### DISPLAY IMAGE #########################

        object_lvis = ['bed', 'stool', 'towel', 'fireplace', 'picture', 'cabinet', 'toilet', 'curtain', 'lighting', 'table', 
            'shelving', 'mirror', 'sofa', 'cushion', 'bathtub', 'chair', 'chest_of_drawers', 'sink', 'seating', 'tv_monitor']

        # Create a blank image for the legend
        legend_height = 480
        legend_width = 640
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

        # Define your color legend values and colors
        legend_values = object_lvis
        legend_colors = palette[0:len(object_lvis)]

        # Calculate the width of each color block
        block_height = legend_height // len(legend_values)

        # Fill the legend with colored blocks and labels
        for i, (value, color) in enumerate(zip(legend_values, legend_colors)):
            start_x = i * block_height
            end_x = (i + 1) * block_height
            legend[start_x:end_x, :, :] = color
            cv2.putText(legend, value, (legend_height // 2, start_x + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if visualise:
            cv2.namedWindow(fig_name, cv2.WINDOW_NORMAL)
            cv2.imshow(fig_name, max_indices_colour)
            cv2.imshow('legend', legend)
            cv2.waitKey(0)

        return max_indices.reshape(-1)
    
    def forward(self, x, memory, proj_indices, observations, sequence_name=None):
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

        # calculate 'memory' simply by projecting the current features into the memory space
        # this allows us to investigate the information lost in the projection process
        if self.merge_type == 'image_projection':
            # create memory features only using the image features - these then get projected back into the image via egocentric memory
            image_features = results[0]
            proj = proj_indices[0]

            # get image features into the format to project to memory
            # upsample image features to the shape of proj
            image_features = F.interpolate(image_features, size=(proj.shape[0], proj.shape[1]), mode="bilinear", align_corners=True)
            image_features = image_features[:,:,::8,::8]
            # project the image features into a new space using 1x1 layer
            image_features = self.map_merge_forward_projection(image_features).squeeze(0)

            # flatten the image features and proj and reduce the number of projection indices
            image_features = image_features.permute(1, 2, 0).reshape(-1, 256)
            proj = proj[::8,::8].reshape(-1)

            # generate the projection matrix
            proj_matrix = torch.cuda.BoolTensor(proj.shape[0], memory[0].shape[0]).fill_(False)
            proj_matrix[torch.arange(proj_matrix.shape[0]), proj] = True
            proj_matrix = proj_matrix.t()

            # only keep the rows (memory elements) that are used
            observed_mem = torch.any(proj_matrix, dim=1)
            proj_matrix = proj_matrix[observed_mem]
            
            # convert to half for matrix multiplication
            proj_matrix = proj_matrix.to(torch.float32)

            # multiply projection and image features
            # this operation can lead to inf values due to overflow on float16
            with autocast(enabled=False):
                sum_image_feat = torch.matmul(proj_matrix, image_features.to(torch.float32))
            # get the max value in the entire array
            count = torch.sum(proj_matrix, dim=1).unsqueeze(1)
            mean = sum_image_feat/count

            # replace memory with zeros
            memory = [torch.zeros_like(memory[0]).cuda()]

            # directly place the memory features into memory
            memory[0][observed_mem] = mean.to(memory[0].dtype)

        # calculate the egocentric memory features
        start = time.time()
        egocentric_memory = []
        for i in range(len(memory)):
            # project memory features into the image frame
            ego_mem_i = memory[i][proj_indices[i]].permute(2, 0, 1).unsqueeze(0).cuda()
            
            # validate the projection indices if sequence name is passed to the function
            if sequence_name and i == 0:
                self.validate_projection_indices(ego_mem_i, sequence_name)

            # downsample the memory features to the size of the highest resolution feature map 
            # ego_mem_i = F.interpolate(ego_mem_i.to(torch.float32), scale_factor=0.125, mode="bilinear")
            ego_mem_i = F.avg_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
            # ego_mem_i = F.max_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
            ego_mem_i = ego_mem_i.squeeze(0)

            # add to the list of memory features
            egocentric_memory.append(ego_mem_i)
        egocentric_memory = torch.stack(egocentric_memory)

        # #code for visualising the memory features
        # def mouse_callback(event, x, y, flags, param):
        #     x_ = int(x/4)
        #     y_ = int(y/4)
        #     if event == cv2.EVENT_LBUTTONDOWN:

        #         # print('Image features on clicked pixel')
        #         # print(results[0][0, :10, int(y_/2), int(x_/2)])

        #         # get the proj index at this location
        #         proj_index = proj_indices[0][y, x]
        #         show_im = show_im_old.copy()

        #         # get the memory embedding at this location
        #         # embedding = memory[0][proj_index][0:10]
        #         # print('memory embedding: ')
        #         # print(embedding)

        #         # memory embedding projected into the frame of the image and downsampled
        #         embedding = egocentric_memory[0, :10, y_, x_]
        #         print('egocentric memory embedding')
        #         print(embedding)

        #         # upsample image features to the shape of proj
        #         proj = proj_indices[0]
        #         image_features = results[0]
        #         image_features = F.interpolate(image_features, size=(proj.shape[0], proj.shape[1]), mode="bilinear", align_corners=True)
        #         image_features = image_features.squeeze(0).permute(1, 2, 0)

        #         # reduce the number of features to make calculation tractable
        #         image_features = image_features[::8,::8,:]
        #         proj = proj[::8,::8]

        #         # colour all points with this index on the image
        #         img_features = []
        #         for u in range(proj.shape[0]):
        #             for v in range(proj.shape[1]):
        #                 if proj[u, v] == proj_index:
        #                     show_im[u*8-4:u*8+4, v*8-4:v*8+4] = [-1, -1, 2]
        #                     # show_im[u, v] = [-1, -1, 2]
        #                     img_features.append(image_features[int(u), int(v), :])
                
        #         # print the mean of the features, this should be the same as the memory embedding above
        #         # img_features = torch.stack(img_features)
        #         # print('Mean Image features: ')
        #         # print(torch.mean(img_features, dim=0)[:10])
                
        #         cv2.imshow("Image", show_im)
        #         key = cv2.waitKey(1) & 0xFF
        #         if key == ord("q"):
        #             pass

        # # display the image
        # show_im = np.transpose(x[0].cpu().numpy(), (1,2,0))
        # show_im_old = show_im.copy()
        
        # # add the projection_indices to the image by colouring it
        # if self.merge_type in ['semantic_gt', 'map_gt']:
        #     seg_classes = proj_indices[0].cpu().numpy()
        #     for u in range(seg_classes.shape[0]):
        #         for v in range(seg_classes.shape[1]):
        #             if seg_classes[u, v]:
        #                 show_im[u, v] = palette[seg_classes[u, v]]/255

        # cv2.imshow("Image", show_im)
        # cv2.setMouseCallback("Image", mouse_callback)

        # # Main loop
        # while True:
        #     cv2.imshow("Image", show_im)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord("q"):
        #         break

        # # Close the OpenCV window
        # cv2.destroyAllWindows()

        # use a gru to include precomputed features
        if self.merge_type == 'gru_precomputed':
            print('using gru to merge precomputed features')
            new_results = []
            for i, res in enumerate(results):
                b, d, w, h = res.shape
                res_dtype = res.dtype

                # downsample the memory features to be used next iterations
                egocentric_memory = F.avg_pool2d(egocentric_memory.to(torch.float32), kernel_size=2, stride=2).to(torch.half)

                # print min and max of res and mem
                # print('res mean: ', torch.mean(res.abs()))

                # flatten and apply the rnn to the resulting sequence
                res_flat = res.permute(0, 2, 3, 1).reshape(-1, 256)
                mem_flat = egocentric_memory.to(torch.half).permute(0, 2, 3, 1).reshape(-1, 256)
                # use autocast() to balance use of float32 and half
                with autocast():
                    new_res = self.merge_rnns[i](mem_flat.cuda(), res_flat)

                # change = (new_res-res_flat).abs()
                # print the average change in feautures
                # print('change due to memory features: ', torch.mean(change))

                # reshape and add to new results
                new_res = new_res.reshape(b, w, h, d).permute(0, 3, 1, 2)
                new_results.append(new_res.to(res_dtype))

            results = new_results
        
        # use cnn projections to include precomputed features
        elif self.merge_type in ['cnn_precomputed', 'semantic_gt', 'map_gt', 'implicit_memory', 'image_projection']:
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
                mem = self.merge_projections[i](egocentric_memory.to(torch.float32).cuda())
                
                # the scaling factor is to balance the magnitude of the two feature spaces - this can be done formally by calculating the L2 dist of the features
                # performance is sensitive to this parameter it seems - balancing memory and image features is important
                if self.merge_type in ['semantic_gt', 'map_gt']:
                    mem = mem*self.map_feature_weight
                if self.merge_type in ['implicit_memory']:
                    # scale up by 10 for conv based memory update
                    # mem = mem*10
                    # if we are using rnn update, scale up by 100 instead
                    # mem = mem*100
                    mem = mem*self.memory_feature_weight
                
                # print min and max of the resulting memory vector
                # print('mem mean: ', torch.mean(mem.abs()))

                # merge by summing the two feature spaces
                if self.feat_fusion == 'sum':
                    new_res = mem+res
                elif self.feat_fusion == 'mem_only':
                    new_res = mem
                elif self.feat_fusion == 'ave':
                    new_res = (mem+res)*0.5
                elif self.feat_fusion == 'image_only':
                    new_res = res

                # change = (new_res-res).abs()
                # print the average change in feautures
                # print('Distance between features: ', torch.mean(change))

                # add to new results
                new_results.append(new_res.to(res_dtype))

            start = time.time()

            ########################## UPDATE THE MEMORY FEATURES ################################

            if self.merge_type == 'implicit_memory':
                
                # define the memory variables
                image_features = results[0]
                proj = proj_indices[0]

                # get image features into the format to project to memory
                # upsample image features to the shape of proj
                image_features = F.interpolate(image_features, size=(proj.shape[0], proj.shape[1]), mode="bilinear", align_corners=True)
                image_features = image_features[:,:,::8,::8]
                # project the image features into a new space using 1x1 layer
                image_features = self.map_merge_memory_projection(image_features).squeeze(0)

                # flatten the image features and proj and reduce the number of projection indices
                image_features = image_features.permute(1, 2, 0).reshape(-1, 256)
                proj = proj[::8,::8].reshape(-1)

                # generate the projection matrix
                proj_matrix = torch.cuda.BoolTensor(proj.shape[0], memory[0].shape[0]).fill_(False)
                proj_matrix[torch.arange(proj_matrix.shape[0]), proj] = True
                proj_matrix = proj_matrix.t()

                # only keep the rows (memory elements) that are used
                observed_mem = torch.any(proj_matrix, dim=1)
                proj_matrix = proj_matrix[observed_mem]
                
                # convert to half for matrix multiplication
                proj_matrix = proj_matrix.to(torch.float32)

                # multiply projection and image features
                # this operation can lead to inf values due to overflow on float16
                with autocast(enabled=False):
                    sum_image_feat = torch.matmul(proj_matrix, image_features.to(torch.float32))
                # get the max value in the entire array
                count = torch.sum(proj_matrix, dim=1).unsqueeze(1)
                mean = sum_image_feat/count

                # replace memory with zeros, and directly place the projected features into memory
                memory = [torch.zeros_like(memory[0]).cuda()]
                memory[0][observed_mem] = mean.to(memory[0].dtype)

                # update the memory features using an rnn
                # with autocast():
                #     memory[0][observed_mem] = self.map_merge_rnn(mean.cuda(), memory[0][observed_mem])

                # update the memory features by concatenating and then projecting with a 1x1 conv
                # mem_update = torch.cat((mean.cuda(), memory[0][observed_mem]), dim=1)

                # mem_update = self.map_merge_memory_update(mem_update.permute(1,0).unsqueeze(0))

                # memory[0][observed_mem] = mem_update.squeeze(0).permute(1,0).to(memory[0].dtype)
                
                # print('Time to update memory features: ', time.time() - start)

            # update the results
            results = new_results

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
        return {f: res for f, res in zip(self._out_features, results)}, memory[0]
    
class CustomRecurrentFPN(FPN):
    def __init__(self, **kwargs):
        self.merge_type = kwargs.pop('merge_type')
        self.norm = kwargs['norm']
        self.feat_fusion = kwargs.pop('fusion')
        self.memory_type = kwargs.pop('memory_type')
        self.memory_feature_weight = kwargs.pop('memory_feature_weight')
        self.map_feature_weight = kwargs.pop('map_feature_weight')
        super().__init__(**kwargs)

        # dimension of extracted image features
        ego_feat_dim = 256
        # dimension of memory features
        mem_feat_dim = 256
        self.memory_dim = mem_feat_dim

        # intialise memory
        self.backbone_type = "recurrent"
        self.memory = None
        self.proj_indices = None
        self.results = None

        # additional models defined here to merge sequences
        if self.merge_type == 'gru_precomputed':
            self.map_merge_gru1 = nn.GRUCell(mem_feat_dim, ego_feat_dim, bias=True)
            self.map_merge_gru2 = nn.GRUCell(mem_feat_dim, ego_feat_dim, bias=True)
            self.map_merge_gru3 = nn.GRUCell(mem_feat_dim, ego_feat_dim, bias=True)
            self.merge_rnns = [self.map_merge_gru1, self.map_merge_gru2, self.map_merge_gru3]

            # change default LSTM initialization
            for rnn in self.merge_rnns:
                noise = 0.01
                rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(rnn.weight_hh)
                rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(rnn.weight_ih)
                rnn.bias_hh.data = torch.zeros_like(rnn.bias_hh)  # redundant with bias_ih
                rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(rnn.bias_ih)
        
        elif self.merge_type in ['cnn_precomputed', 'implicit_memory', 'image_projection']:
            merge_norm = get_norm(self.norm, ego_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_projection4 = Conv2d(
                mem_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection5 = Conv2d(
                mem_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection6 = Conv2d(
                mem_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )

            self.merge_recurrent_projections = [self.map_merge_projection4, self.map_merge_projection5, self.map_merge_projection6]

        if self.merge_type in ['semantic_gt', 'map_gt'] or self.memory_type in ['implicit_memory', 'explicit_map', 'explicit_map_ignore']:
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

        # layers for updating the memory features
        if self.merge_type == 'implicit_memory':
            # use the FPNSemSeg head to merge image features
            input_shape = {k: v for k, v in self.output_shape().items() if k in ['p3', 'p4', 'p5']}
            self.map_merge_semseg_head = CustomSemSegFPNHead(input_shape=input_shape, num_classes=15, conv_dims=256, common_stride=4)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            merge_norm = get_norm(self.norm, mem_feat_dim)
            self.map_merge_memory_projection = Conv2d(
                ego_feat_dim, 256, kernel_size=1, bias=True, norm=merge_norm
            )

            # a 1x1 convolution in 1 dimension to reduce combine memory and image features and project them to a new representation space
            # self.map_merge_memory_update = torch.nn.Conv1d(512, 256, kernel_size=1)

            # # an rnn to update the memory features
            self.map_merge_rnn = nn.GRUCell(256, mem_feat_dim, bias=True)

            # # # change default LSTM initialization
            noise = 0.01
            self.map_merge_rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.weight_hh)
            self.map_merge_rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.weight_ih)
            self.map_merge_rnn.bias_hh.data = torch.zeros_like(self.map_merge_rnn.bias_hh)  # redundant with bias_ih
            self.map_merge_rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.bias_ih)

        # layers for updating the memory features using roi features
        if self.merge_type == 'object_memory':
            self.memory_dim = 256
            merge_norm = get_norm(self.norm, ego_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_projection1 = Conv2d(
                self.memory_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection2 = Conv2d(
                self.memory_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )
            self.map_merge_projection3 = Conv2d(
                self.memory_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )

            self.merge_recurrent_projections = [self.map_merge_projection1, self.map_merge_projection2, self.map_merge_projection3]
            # a 1x1 convolution in 1 dimension to reduce combine memory and image features and project them to a new representation space
            # self.map_merge_memory_update = torch.nn.Conv1d(512, 256, kernel_size=1)

            # # an rnn to update the memory features
            self.map_merge_rnn = nn.GRUCell(512, self.memory_dim, bias=True)

            # # # change default LSTM initialization
            noise = 0.01
            self.map_merge_rnn.weight_hh.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.weight_hh)
            self.map_merge_rnn.weight_ih.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.weight_ih)
            self.map_merge_rnn.bias_hh.data = torch.zeros_like(self.map_merge_rnn.bias_hh)  # redundant with bias_ih
            self.map_merge_rnn.bias_ih.data = -noise + 2 * noise * torch.rand_like(self.map_merge_rnn.bias_ih)
            
        # layers for the image_projection test
        if self.merge_type == 'image_projection':
            merge_norm  = get_norm(self.norm, ego_feat_dim)

            # a series of 1x1 convolutions that use a linear projection to map memory features into the space of the ego features
            self.map_merge_forward_projection = Conv2d(
                ego_feat_dim, ego_feat_dim, kernel_size=1, bias=True, norm=merge_norm
            )

    def reset_memory(self, memory_size, zs_weight=None):
        self.memory = [torch.zeros(memory_size, self.memory_dim, dtype=torch.half).cuda()]
        self.zs_weight = zs_weight

    def inference_with_proposals(self, proposals, zs_weight, thresh):
        # get the box features, then convert to scores
        box_features = proposals[0].__getattr__('feat')
        boxes = proposals[0].__getattr__('proposal_boxes').tensor
        masks = proposals[0].pred_masks

        # get the proposal scores
        if len(proposals) > 0 and proposals[0].has('scores'):
            proposal_scores = [p.get('scores') for p in proposals]  
        else:
            proposal_scores = [p.get('objectness_logits') for p in proposals]   

        # get the indices of the proposals that have a score < 1 - this occurs during training when gt boxes are added to the proposals
        proposal_indices = torch.where(proposal_scores[0] < 1)[0]
        # keep all proposal scores that have a score < 1
        proposal_scores = [proposal_scores[0][proposal_indices]]
        # keep all boxes that have a score < 1
        boxes = boxes[proposal_indices, :]
        masks = masks[proposal_indices, ...]
        box_features = box_features[proposal_indices, :]    

        # calculate scores from clip embeddings and run faster_rcnn_inference to get post nms results
        # normalise the box features
        box_features = 50.0 * torch.nn.functional.normalize(box_features, p=2, dim=1)
        # dot product with clip embeddings to get logits
        box_scores = torch.mm(box_features, zs_weight)
        
        # convert logits to probabilities by taking sigmoid and combining with proposal scores
        box_scores = [box_scores.sigmoid()]
        # box scores are calculated by combining the clip embeddings classification with the proposal scores
        box_scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(box_scores, proposal_scores)]

        # run faster_rcnn_inference to get post nms results - the result is similar to detic inference if we change it to use scores only from the first cascade head
        test_score_thresh = thresh
        test_nms_thresh = 0.5
        test_topk_per_image = 100
        mask_thresh = 0.5
        pred_instances, proposal_indices = fast_rcnn_inference(
            (boxes,),
            box_scores,
            [(480,640)],
            test_score_thresh,
            test_nms_thresh,
            test_topk_per_image,
        )
        
        # no memory update if there are no proposals
        if len(pred_instances[0]) == 0:
            return

        proposal_indices = torch.unique(proposal_indices[0])
        # given the post nms result we have a set of boxes, masks and features
        boxes = boxes[proposal_indices, :]
        box_features = box_features[proposal_indices, :]
        masks = masks[proposal_indices, ...].squeeze(1)
        masks = paste_masks_in_image(masks, boxes, (480, 640), threshold=mask_thresh)

        return boxes, box_features, masks, pred_instances

    def box_to_image_features(self, box_features, masks):
        # egocentric features start as an empty tensor of shape (1, 512, 480, 640)
        image_features = torch.cuda.FloatTensor(1, 512, 480, 640).fill_(0)
        observations = torch.cuda.FloatTensor(1, 1, 480, 640).fill_(0)

        # we use the masks to populate the egocentric features
        for i in range(box_features.shape[0]):
            mask = masks[i, :, :]
            feature = box_features[i, :].reshape(1, 512)

            image_features[:, :, mask] += feature.unsqueeze(2).repeat(1,1,torch.sum(mask).item())
            observations[:, :, mask] += 1

        # calculate the average features, but only for pixels where observation > 0
        observed_pixels = (observations>0).squeeze(0).squeeze(0)
        image_features[:,:, observed_pixels] = image_features[:, :, observed_pixels]/observations[:,:,observed_pixels]

        return image_features, observed_pixels
    
    def project_image_features(self, image_features, observed_pixels):
        # extract only those image features and projection indices that have been observed, and reshape to flat tensors
        image_features = image_features[:,:,observed_pixels].squeeze(0)
        image_features = image_features.permute(1,0).reshape(-1, 512)
        proj = self.proj_indices[0][observed_pixels]

        # project the image features into a new space using 1x1 layer
        # image_features = self.map_merge_memory_projection(image_features).squeeze(0)
        
        # downsample each tensor to be the same size
        proj = proj[::8]
        image_features = image_features[::8]

        # generate the projection matrix
        proj_matrix = torch.cuda.BoolTensor(proj.shape[0], self.memory[0].shape[0]).fill_(False)
        proj_matrix[torch.arange(proj_matrix.shape[0]), proj] = True
        proj_matrix = proj_matrix.t()

        # only keep the rows (memory elements) that are used
        observed_mem = torch.any(proj_matrix, dim=1)
        proj_matrix = proj_matrix[observed_mem]
        
        # convert to half for matrix multiplication
        proj_matrix = proj_matrix.to(torch.float32)

        # multiply projection and image features
        # this operation can lead to inf values due to overflow on float16
        with autocast(enabled=False):
            sum_image_feat = torch.matmul(proj_matrix, image_features.to(torch.float32))
        # get the max value in the entire array
        count = torch.sum(proj_matrix, dim=1).unsqueeze(1)
        mean = sum_image_feat/count

        return mean, observed_mem

    def update_memory(self, proposals = None, zs_weight=None, frame=None):
        start = time.time()

        ########################## UPDATE THE MEMORY FEATURES ################################

        if self.merge_type == 'implicit_memory':
            
            # define the memory variables
            proj = self.proj_indices[0]

            # get image features into the format to project to memory
            image_features = self.results[0]
            image_features = F.interpolate(image_features, size=(proj.shape[0], proj.shape[1]), mode="bilinear", align_corners=True)

            # alternatively, we want to merge the three layers of the feature pyramid as per panoptic fpn
            # image_features = {f: res for f, res in zip(self._out_features[0:3], results)}
            # image_features, _ = self.map_merge_semseg_head(image_features)

            # downsample the image features
            image_features = image_features[:,:,::8,::8]
            # project the image features into a new space using 1x1 layer
            image_features = self.map_merge_memory_projection(image_features).squeeze(0)

            # flatten the image features and proj and reduce the number of projection indices
            image_features = image_features.permute(1, 2, 0).reshape(-1, 256)
            proj = proj[::8,::8].reshape(-1)

            # generate the projection matrix
            proj_matrix = torch.cuda.BoolTensor(proj.shape[0], self.memory[0].shape[0]).fill_(False)
            proj_matrix[torch.arange(proj_matrix.shape[0]), proj] = True
            proj_matrix = proj_matrix.t()

            # only keep the rows (memory elements) that are used
            observed_mem = torch.any(proj_matrix, dim=1)
            proj_matrix = proj_matrix[observed_mem]
            
            # convert to half for matrix multiplication
            proj_matrix = proj_matrix.to(torch.float32)

            # multiply projection and image features
            # this operation can lead to inf values due to overflow on float16
            with autocast(enabled=False):
                sum_image_feat = torch.matmul(proj_matrix, image_features.to(torch.float32))
            # get the max value in the entire array
            count = torch.sum(proj_matrix, dim=1).unsqueeze(1)
            mean = sum_image_feat/count

            # replace memory with zeros, and directly place the projected features into memory
            # self.memory = [torch.zeros_like(self.memory[0]).cuda()]
            # self.memory[0][observed_mem] = mean.to(self.memory[0].dtype)

            # update the memory features using an rnn
            with autocast():
                self.memory[0][observed_mem] = self.map_merge_rnn(mean.cuda(), self.memory[0][observed_mem])

            # update the memory features by concatenating and then projecting with a 1x1 conv
            # mem_update = torch.cat((mean.cuda(), self.memory[0][observed_mem]), dim=1)
            # mem_update = self.map_merge_memory_update(mem_update.permute(1,0).unsqueeze(0))
            # self.memory[0][observed_mem] = mem_update.squeeze(0).permute(1,0).to(self.memory[0].dtype)
            
            # print('Time to update memory features: ', time.time() - start)

        elif self.merge_type == 'object_memory':

            # run inference with proposals to get masks and classification scores
            boxes, box_features, masks, pred_instances = self.inference_with_proposals(proposals, zs_weight, 0.3)
            
            # # display the image with the masks and boxes overlaid
            # image = frame['image'].permute(1, 2, 0).cpu().numpy()
            # v = Visualizer(image, None)
            # out = v.overlay_instances(
            #     masks=masks.cpu().numpy(),
            #     boxes=boxes.cpu().numpy(),
            #     alpha=0.6,
            #     )
            # cv2.imshow('image', out.get_image()[:, :, ::-1])
            # # cv2.waitKey(0)
            
            # project the box features into the image frame
            image_features, observed_pixels = self.box_to_image_features(box_features, masks)

            # visualise the output
            # self.visualise_clip_image_features(image_features, zs_weight)
        
            mean, observed_mem = self.project_image_features(image_features, observed_pixels)

            # return if there are no observed memory elements
            if mean.shape[0] == 0 or self.memory[0][observed_mem].shape[0] == 0:
                return     

            # replace memory with zeros, and directly place the projected features into memory
            # self.memory = [torch.zeros_like(self.memory[0]).cuda()]
            # self.memory[0][observed_mem] = mean.to(self.memory[0].dtype)

            # update the memory features using an rnn
            with autocast():
                self.memory[0][observed_mem] = self.map_merge_rnn(mean.cuda(), self.memory[0][observed_mem])

            # update the memory features by concatenating and then projecting with a 1x1 conv
            # mem_update = torch.cat((mean.cuda(), self.memory[0][observed_mem]), dim=1)
            # mem_update = self.map_merge_memory_update(mem_update.permute(1,0).unsqueeze(0))
            # self.memory[0][observed_mem] = mem_update.squeeze(0).permute(1,0).to(self.memory[0].dtype)
            
            # print('Time to update memory features: ', time.time() - start)

    def visualise_clip_image_features(self, image_features, zs_weight, fig_name = 'image_features', mask=None, thresh = 0.4, visualise = True):
        
        ################### USE FEATURES TO GET CLASSIFICATIONS ######################3
        
        # get the shape
        _, c, w, h = image_features.shape
        
        # flatten features    
        flat_features = image_features.squeeze(0).permute(1,2,0).reshape(-1, 512)
        
        # normalise the box features
        norm_feautures = 50.0 * torch.nn.functional.normalize(flat_features, p=2, dim=1)

        # check if norm_features contains none
        if torch.any(torch.isnan(norm_feautures)):
            norm_features = flat_features
        
        # dot product with clip embeddings to get logits
        image_feat_scores = torch.mm(norm_feautures.to(torch.half), zs_weight.to(torch.half))[:,:20]
        
        # apply softmax
        image_feat_scores = image_feat_scores.softmax(dim=1)
        
        # get the maximum score for each pixel
        max_scores, max_indices = torch.max(image_feat_scores, dim=1)

        # reshape the max indices and the scores
        max_indices = max_indices.reshape(w, h).cpu().numpy()
        max_scores = max_scores.reshape(w, h, 1)

        # # convert to rgb for displaying the results
        # max_indices_colour = palette[max_indices]

        ################# APPLY THRESHOLD TO EITHER MASK OR SCORES #################

        # use the input mask to threshold the results 
        if mask is not None:
            # max_scores = max_scores * mask.reshape(w,h,1)
            max_scores = mask.reshape(w,h,1)

        # scale the colour by the score for visualisation
        # max_indices = max_indices * max_scores.cpu().numpy()
        
        # alternatively, generate an explicit map by apply a threshold
        max_indices[max_scores.squeeze(2).cpu().numpy() < thresh] = -1

        # convert to rgb to display the results
        max_indices_colour = palette[max_indices]

        # convert to 32 bit signed int
        max_indices_colour = max_indices_colour.astype(np.uint8)

        ################### DISPLAY IMAGE #########################

        object_lvis = ['bed', 'stool', 'towel', 'fireplace', 'picture', 'cabinet', 'toilet', 'curtain', 'lighting', 'table', 
            'shelving', 'mirror', 'sofa', 'cushion', 'bathtub', 'chair', 'chest_of_drawers', 'sink', 'seating', 'tv_monitor']

        # Create a blank image for the legend
        legend_height = 480
        legend_width = 640
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

        # Define your color legend values and colors
        legend_values = object_lvis
        legend_colors = palette[0:len(object_lvis)]

        # Calculate the width of each color block
        block_height = legend_height // len(legend_values)

        # Fill the legend with colored blocks and labels
        for i, (value, color) in enumerate(zip(legend_values, legend_colors)):
            start_x = i * block_height
            end_x = (i + 1) * block_height
            legend[start_x:end_x, :, :] = color
            cv2.putText(legend, value, (legend_height // 2, start_x + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if visualise:
            cv2.namedWindow(fig_name, cv2.WINDOW_NORMAL)
            cv2.imshow(fig_name, max_indices_colour)
            cv2.imshow('legend', legend)
            cv2.waitKey(0)

        return max_indices.reshape(-1)
        

    def visualise_memory(self, x, proj_indices, egocentric_memory, results):    
        def mouse_callback(event, x, y, flags, param):
            x_ = int(x/4)
            y_ = int(y/4)
            if event == cv2.EVENT_LBUTTONDOWN:

                # print('Image features on clicked pixel')
                # print(results[0][0, :10, int(y_/2), int(x_/2)])

                # get the proj index at this location
                proj_index = proj_indices[0][y, x]
                show_im = show_im_old.copy()

                # get the memory embedding at this location
                # embedding = memory[0][proj_index][0:10]
                # print('memory embedding: ')
                # print(embedding)

                # memory embedding projected into the frame of the image and downsampled
                embedding = egocentric_memory[0, :10, y_, x_]
                print('egocentric memory embedding')
                print(embedding)

                # upsample image features to the shape of proj
                proj = proj_indices[0]
                image_features = results[0]
                image_features = F.interpolate(image_features, size=(proj.shape[0], proj.shape[1]), mode="bilinear", align_corners=True)
                image_features = image_features.squeeze(0).permute(1, 2, 0)

                # reduce the number of features to make calculation tractable
                image_features = image_features[::8,::8,:]
                proj = proj[::8,::8]

                # colour all points with this index on the image
                img_features = []
                for u in range(proj.shape[0]):
                    for v in range(proj.shape[1]):
                        if proj[u, v] == proj_index:
                            show_im[u*8-4:u*8+4, v*8-4:v*8+4] = [-1, -1, 2]
                            # show_im[u, v] = [-1, -1, 2]
                            img_features.append(image_features[int(u), int(v), :])
                
                # print the mean of the features, this should be the same as the memory embedding above
                # img_features = torch.stack(img_features)
                # print('Mean Image features: ')
                # print(torch.mean(img_features, dim=0)[:10])
                
                cv2.imshow("Image", show_im)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    pass

        # display the image
        show_im = np.transpose(x[0].cpu().numpy(), (1,2,0))
        show_im_old = show_im.copy()
        
        # add the projection_indices to the image by colouring it
        if self.merge_type in ['semantic_gt', 'map_gt']:
            seg_classes = proj_indices[0].cpu().numpy()
            for u in range(seg_classes.shape[0]):
                for v in range(seg_classes.shape[1]):
                    if seg_classes[u, v]:
                        show_im[u, v] = palette[seg_classes[u, v]]/255

        cv2.imshow("Image", show_im)
        cv2.setMouseCallback("Image", mouse_callback)

        # Main loop
        while True:
            cv2.imshow("Image", show_im)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Close the OpenCV window
        cv2.destroyAllWindows()
 
    def validate_projection_indices(self, ego_mem_i, sequence_name):    
        # load the memory features from file
        h5file = h5py.File(os.path.join('../Semantic-MapNet/data/training/', 'smnet_training_data_memory', sequence_name[0:13]) + '.h5', 'r')
        map_memory = np.array(h5file['memory'])
        semmap = np.array(h5file['pred_map'])
        map_memory = np.transpose(map_memory[0], (1, 2, 0))
        h5file.close()

        # load the raw projection information
        h5file = h5py.File(os.path.join('../Semantic-MapNet/data/training/', 'smnet_training_data', sequence_name), 'r')
        projection_indices = np.array(h5file['projection_indices'], dtype=np.float32)
        masks_outliers = np.array(h5file['masks_outliers'], dtype=np.bool)
        sensor_positions = np.array(h5file['sensor_positions'], dtype=np.float32)
        h5file.close()

        # load scene information
        resolution = 0.02 # topdown resolution'
        semmap_info = json.load(open('../Semantic-MapNet/data//semmap_GT_info.json', 'r'))
        map_world_shift = np.array(semmap_info[sequence_name[0:13]]['map_world_shift'])

        # transfer to Pytorch
        projection_indices = torch.FloatTensor(projection_indices)
        masks_outliers = torch.BoolTensor(masks_outliers)
        sensor_positions = torch.FloatTensor(sensor_positions)
        map_world_shift = torch.FloatTensor(map_world_shift)
        projection_indices -= map_world_shift
        pixels_in_map = (projection_indices[:,:,:, [0,2]] / resolution).round().long()

        # get a list of all grid indices that are viewed in this sequence
        pixels_in_map = pixels_in_map.numpy()

        # project ego_memory features back to the image 
        predicted_mem = ego_mem_i.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_mem = map_memory[pixels_in_map[0, :, :, 1], pixels_in_map[0, :, :, 0], :]
        print('Calculated memory features: ', predicted_mem.shape)
        print('Ground truth memory features: ', gt_mem.shape)

        # check it is the same as the original memory features
        # print(predicted_mem[0, 0, :])
        # print(gt_mem[0, 0, :])
        # flatten both arrays
        predicted_mem_flat = predicted_mem.reshape(-1, 256)
        gt_mem_flat = gt_mem.reshape(-1, 256)
        # calculate the distance between each corresponding element in the arrays
        dist = np.linalg.norm(predicted_mem_flat - gt_mem_flat, axis=1)
        # print the mean of the distance
        print('Mean distance between calculated and ground truth memory features: ', np.mean(dist))
        # for comparison, randomly shuffle the memory features and repeat the calculation
        np.random.shuffle(predicted_mem_flat)
        dist = np.linalg.norm(predicted_mem_flat - gt_mem_flat, axis=1)
        print('Mean distance between random features ', np.mean(dist))
        
        # assess distance between features
        # map_w, map_h = map_memory.shape[0], map_memory.shape[1]
        # map_mem_flat = map_memory.reshape(-1, 256)
        # predicted_mem_flat = predicted_mem.reshape(-1, 256)
        # from scipy.spatial.distance import cdist
        # dist = cdist(map_mem_flat, predicted_mem_flat, 'euclidean')
        # closest_indices = np.argmin(dist, axis=1)
        # # convert flattened indices to 2D indices
        # closest_indices = np.unravel_index(closest_indices, (map_w, map_h))
        # closest_indices = closest_indices.reshape(480, 640, 2)
        # print(closest_indices)
        # print(pixels_in_map[0])
    
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

        ##################### ADD VARIABLES REQUIRED FOR MEMORY UPDATE #####################
        self.results = results
        self.proj_indices = proj_indices
        
        ########################## MERGE IN THE MEMORY FEATURES ################################

        # calculate 'memory' simply by projecting the current features into the memory space
        # this allows us to investigate the information lost in the projection process
        if self.merge_type == 'image_projection':
            # create memory features only using the image features - these then get projected back into the image via egocentric memory
            image_features = results[0]
            proj = proj_indices[0]

            # get image features into the format to project to memory
            # upsample image features to the shape of proj
            image_features = F.interpolate(image_features, size=(proj.shape[0], proj.shape[1]), mode="bilinear", align_corners=True)
            image_features = image_features[:,:,::8,::8]
            # project the image features into a new space using 1x1 layer
            image_features = self.map_merge_forward_projection(image_features).squeeze(0)

            # flatten the image features and proj and reduce the number of projection indices
            image_features = image_features.permute(1, 2, 0).reshape(-1, 256)
            proj = proj[::8,::8].reshape(-1)

            # generate the projection matrix
            proj_matrix = torch.cuda.BoolTensor(proj.shape[0], self.memory[0].shape[0]).fill_(False)
            proj_matrix[torch.arange(proj_matrix.shape[0]), proj] = True
            proj_matrix = proj_matrix.t()

            # only keep the rows (memory elements) that are used
            observed_mem = torch.any(proj_matrix, dim=1)
            proj_matrix = proj_matrix[observed_mem]
            
            # convert to half for matrix multiplication
            proj_matrix = proj_matrix.to(torch.float32)

            # multiply projection and image features
            # this operation can lead to inf values due to overflow on float16
            with autocast(enabled=False):
                sum_image_feat = torch.matmul(proj_matrix, image_features.to(torch.float32))
            # get the max value in the entire array
            count = torch.sum(proj_matrix, dim=1).unsqueeze(1)
            mean = sum_image_feat/count

            # replace memory with zeros
            self.memory = [torch.zeros_like(self.memory[0]).cuda()]

            # directly place the memory features into memory
            self.memory[0][observed_mem] = mean.to(self.memory[0].dtype)

        # calculate the egocentric memory features
        start = time.time()
        egocentric_memory = []
        for i in range(len(self.memory)):
            # project memory features into the image frame
            ego_mem_i = self.memory[i][proj_indices[i]].permute(2, 0, 1).unsqueeze(0).cuda()

            # # validate the projection indices if sequence name is passed to the function
            # self.visualise_clip_image_features(ego_mem_i, self.zs_weight)
            # if sequence_name and i == 0:
            #     self.validate_projection_indices(ego_mem_i, sequence_name)

            # downsample the memory features to the size of the highest resolution feature map 
            # ego_mem_i = F.interpolate(ego_mem_i.to(torch.float32), scale_factor=0.125, mode="bilinear")
            ego_mem_i = F.avg_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
            # ego_mem_i = F.max_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
            ego_mem_i = ego_mem_i.squeeze(0)

            # add to the list of memory features
            egocentric_memory.append(ego_mem_i)
        egocentric_memory = torch.stack(egocentric_memory)

        #code for visualising the memory features
        # self.visualise_memory(x, proj_indices, egocentric_memory, results)

        # use a gru to include precomputed features
        if self.merge_type == 'gru_precomputed':
            print('using gru to merge precomputed features')
            new_results = []
            for i, res in enumerate(results):
                b, d, w, h = res.shape
                res_dtype = res.dtype

                # downsample the memory features to be used next iterations
                egocentric_memory = F.avg_pool2d(egocentric_memory.to(torch.float32), kernel_size=2, stride=2).to(torch.half)

                # print min and max of res and mem
                # print('res mean: ', torch.mean(res.abs()))

                # flatten and apply the rnn to the resulting sequence
                res_flat = res.permute(0, 2, 3, 1).reshape(-1, 256)
                mem_flat = egocentric_memory.to(torch.half).permute(0, 2, 3, 1).reshape(-1, 256)
                # use autocast() to balance use of float32 and half
                with autocast():
                    new_res = self.merge_rnns[i](mem_flat.cuda(), res_flat)

                # change = (new_res-res_flat).abs()
                # print the average change in feautures
                # print('change due to memory features: ', torch.mean(change))

                # reshape and add to new results
                new_res = new_res.reshape(b, w, h, d).permute(0, 3, 1, 2)
                new_results.append(new_res.to(res_dtype))

            results = new_results
        
        # use cnn projections to include precomputed features
        elif self.merge_type in ['cnn_precomputed', 'implicit_memory', 'image_projection', 'object_memory']:
            print('enhancing features with pixel based memory')
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
                mem = self.merge_recurrent_projections[i](egocentric_memory.to(torch.float32).cuda())

                # calculate the L2 distance of all non-zero memory features, and use this to scale the memory features
                mem_norm = torch.linalg.vector_norm(mem, dim=1)
                res_norm = torch.linalg.vector_norm(res, dim=1)
                if len(mem_norm[mem_norm != 0]) == 0:
                    scale_factor = 1
                else:
                    scale_factor = torch.mean(res_norm[res_norm != 0])/torch.mean(mem_norm[mem_norm != 0])
                # print(scale_factor)
                mem = mem*self.memory_feature_weight
                
                # the scaling factor is to balance the magnitude of the two feature spaces - this can be done formally by calculating the L2 dist of the features
                # performance is sensitive to this parameter it seems - balancing memory and image features is important
                # if self.merge_type in ['semantic_gt', 'map_gt']:
                #     mem = mem*500
                # if self.merge_type in ['implicit_memory', 'object_memory']:
                #     # scale up by 10 for conv based memory update
                #     # mem = mem*10
                #     # if we are using rnn update, scale up by 100 instead
                
                # print min and max of the resulting memory vector
                # print('mem mean: ', torch.mean(mem.abs()))

                # merge by summing the two feature spaces
                if self.feat_fusion == 'sum':
                    new_res = mem+res
                elif self.feat_fusion == 'mem_only':
                    new_res = mem
                elif self.feat_fusion == 'ave':
                    new_res = (mem+res)*0.5
                elif self.feat_fusion == 'image_only':
                    new_res = res

                # change = (new_res-res).abs()
                # print the average change in feautures
                # print('Distance between features: ', torch.mean(change))

                # add to new results
                new_results.append(new_res.to(res_dtype))
            
            # update the results
            results = new_results

        # use cnn projections to include map based memory features
        if self.merge_type in ['semantic_gt', 'map_gt'] or self.memory_type in ['implicit_memory', 'explicit_map']:
            print('enhancing features with map based memory')
            start = time.time()
            egocentric_memory = []
            for i in range(len(map_memory)):
                # project memory features into the image frame
                ego_mem_i = map_memory[i][proj_indices[i]].permute(2, 0, 1).unsqueeze(0).cuda()
                ego_obs_i = observations[i][proj_indices[i]]

                # # validate the projection indices if sequence name is passed to the function
                # self.visualise_clip_image_features(ego_mem_i, self.zs_weight)
                # if sequence_name and i == 0:
                #     self.validate_projection_indices(ego_mem_i, sequence_name)

                # downsample the memory features to the size of the highest resolution feature map 
                # ego_mem_i = F.interpolate(ego_mem_i.to(torch.float32), scale_factor=0.125, mode="bilinear")
                ego_mem_i = F.avg_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
                # ego_mem_i = F.max_pool2d(ego_mem_i.to(torch.float32), kernel_size=4, stride=4)
                ego_mem_i = ego_mem_i.squeeze(0)

                # add to the list of memory features
                egocentric_memory.append(ego_mem_i)

            egocentric_memory = torch.stack(egocentric_memory)

            # calculate the natural logarithm for scaling
            # observation_scale = torch.log(torch.mean(observations[0]+1))

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

                # calculate the L2 distance of all non-zero memory features, and use this to scale the memory features
                mem_norm = torch.linalg.vector_norm(mem, dim=1)
                res_norm = torch.linalg.vector_norm(res, dim=1)
                if len(mem_norm[mem_norm != 0]) == 0:
                    norm_factor = torch.tensor(0)
                else:
                    norm_factor = torch.mean(res_norm[res_norm != 0])/torch.mean(mem_norm[mem_norm != 0])
                
                # calculate how to scale the memory
                # print('using normalised and observation scaling')
                # scale_factor = norm_factor*observation_scale
                
                # print('using normalised scaling')
                # scale_factor = 4*norm_factor

                # print('using default scaling')
                # scale_factor = torch.tensor(500)
                
                # # clip at 1000
                # if torch.isinf(scale_factor) or scale_factor>1000 or torch.isnan(scale_factor):
                #     scale_factor = torch.tensor(1000)
                
                # scale the mem
                mem = mem*self.map_feature_weight
                # print(scale_factor)
                
                # the scaling factor is to balance the magnitude of the two feature spaces - this can be done formally by calculating the L2 dist of the features
                # performance is sensitive to this parameter it seems - balancing memory and image features is important
                # if self.merge_type in ['semantic_gt', 'map_gt']:
                #     mem = mem*500
                # if self.merge_type in ['implicit_memory', 'object_memory']:
                #     # scale up by 10 for conv based memory update
                #     # mem = mem*10
                #     # if we are using rnn update, scale up by 100 instead
                #     mem = mem*100
                
                # print min and max of the resulting memory vector
                # print('mem mean: ', torch.mean(mem.abs()))

                # merge by summing the two feature spaces
                if self.feat_fusion == 'sum':
                    new_res = mem+res
                elif self.feat_fusion == 'mem_only':
                    new_res = mem
                elif self.feat_fusion == 'ave':
                    new_res = (mem+res)*0.5
                elif self.feat_fusion == 'image_only':
                    new_res = res

                # change = (new_res-res).abs()
                # print the average change in feautures
                # print('Distance between features: ', torch.mean(change))

                # add to new results
                new_results.append(new_res.to(res_dtype))
            
            # update the results
            results = new_results

            # start = time.time()

            # ########################## UPDATE THE MEMORY FEATURES ################################

            # if self.merge_type == 'implicit_memory':
                
            #     # define the memory variables
            #     proj = proj_indices[0]

            #     # get image features into the format to project to memory
            #     image_features = results[0]
            #     image_features = F.interpolate(image_features, size=(proj.shape[0], proj.shape[1]), mode="bilinear", align_corners=True)
                
            #     # alternatively, we want to merge the three layers of the feature pyramid as per panoptic fpn
            #     # image_features = {f: res for f, res in zip(self._out_features[0:3], results)}
            #     # image_features, _ = self.map_merge_semseg_head(image_features)

            #     # downsample the image features
            #     image_features = image_features[:,:,::8,::8]
            #     # project the image features into a new space using 1x1 layer
            #     image_features = self.map_merge_memory_projection(image_features).squeeze(0)

            #     # flatten the image features and proj and reduce the number of projection indices
            #     image_features = image_features.permute(1, 2, 0).reshape(-1, 256)
            #     proj = proj[::8,::8].reshape(-1)

            #     # generate the projection matrix
            #     proj_matrix = torch.cuda.BoolTensor(proj.shape[0], self.memory[0].shape[0]).fill_(False)
            #     proj_matrix[torch.arange(proj_matrix.shape[0]), proj] = True
            #     proj_matrix = proj_matrix.t()

            #     # only keep the rows (memory elements) that are used
            #     observed_mem = torch.any(proj_matrix, dim=1)
            #     proj_matrix = proj_matrix[observed_mem]
                
            #     # convert to half for matrix multiplication
            #     proj_matrix = proj_matrix.to(torch.float32)

            #     # multiply projection and image features
            #     # this operation can lead to inf values due to overflow on float16
            #     with autocast(enabled=False):
            #         sum_image_feat = torch.matmul(proj_matrix, image_features.to(torch.float32))
            #     # get the max value in the entire array
            #     count = torch.sum(proj_matrix, dim=1).unsqueeze(1)
            #     mean = sum_image_feat/count

            #     # replace memory with zeros, and directly place the projected features into memory
            #     # self.memory = [torch.zeros_like(self.memory[0]).cuda()]
            #     # self.memory[0][observed_mem] = mean.to(self.memory[0].dtype)

            #     # update the memory features using an rnn
            #     with autocast():
            #         self.memory[0][observed_mem] = self.map_merge_rnn(mean.cuda(), self.memory[0][observed_mem])

            #     # update the memory features by concatenating and then projecting with a 1x1 conv
            #     # mem_update = torch.cat((mean.cuda(), self.memory[0][observed_mem]), dim=1)
            #     # mem_update = self.map_merge_memory_update(mem_update.permute(1,0).unsqueeze(0))
            #     # self.memory[0][observed_mem] = mem_update.squeeze(0).permute(1,0).to(self.memory[0].dtype)
                
            #     # print('Time to update memory features: ', time.time() - start)



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
        return {f: res for f, res in zip(self._out_features, results)}, self.memory[0]

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

        # add map features here...
        if self.merge_type == 'cnn_precomputed_bottom':
            print('merging in the bottom up backbone: ', self.merge_type)
            # print(x.shape)
            # print(memory.shape)

            b, d, w, h = x.shape
            res_dtype = x.dtype

            # print min and max of res and mem
            # print('x mean: ', torch.mean(x.abs()))

            # apply the 1x1 convolution to the memory features to map to same space as the image features
            mem = self.map_merge_projection(memory.to(torch.float32).cuda())*25
            # print min and max of the resulting memory vector
            # print('mem mean: ', torch.mean(mem.abs()))

            # merge by summing the two feature spaces
            x = mem+x

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