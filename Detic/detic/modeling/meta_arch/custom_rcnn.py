# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import json
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.utils.comm as comm
import random
import math
import h5py
import os
import time

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.layers.mask_ops import paste_masks_in_image


from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds

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

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        **kwargs):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        self.map_conditioned = kwargs.pop('map_conditioned')
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
        })
        ret['map_conditioned'] = cfg.MODEL.TIMM.BASE_NAME == 'resnet50_in21k_map'
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
        
        return ret


    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        
        # display the image
        # for i in range(len(batched_inputs)):
        #     image = images.tensor[i].permute(1, 2, 0).cpu().numpy()
        #     print(images.tensor[i])
        #     print(images.tensor.dtype)
        #     print(batched_inputs[i]['image'])
        #     print(batched_inputs[i]['image'].dtype)
        #     cv2.imshow('image', image)
        #     cv2.imshow('raw_image', batched_inputs[i]['image'].permute(1, 2, 0).cpu().numpy())
        #     cv2.waitKey(0)

        # also need to preprocess memory here....
        if self.map_conditioned:
            # get the memory and projection tensors
            memory, projection = self.preprocess_spatial_memory(batched_inputs)
            # create memory tensor and projection tensor - sequence name added for testing projections
            # features = self.backbone(images.tensor, memory, projection, sequence_name=batched_inputs[0]['sequence_name'])
            features = self.backbone(images.tensor, memory, projection)
        else:
            features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if self.with_image_labels:
            for inst, x in zip(gt_instances, batched_inputs):
                inst._ann_type = x['ann_type']
                inst._pos_category_ids = x['pos_category_ids']
            ann_types = [x['ann_type'] for x in batched_inputs]
            assert len(set(ann_types)) == 1
            ann_type = ann_types[0]
            if ann_type in ['prop', 'proptag']:
                for t in gt_instances:
                    t.gt_classes *= 0
        
        if self.fp16: # TODO (zhouxy): improve
            if self.map_conditioned:
                with autocast():
                    # get the memory and projection tensors
                    memory, projection = self.preprocess_spatial_memory(batched_inputs)
                    # create memory tensor and projection tensor
                    features = self.backbone(images.tensor.half(), memory, projection)
            else:
                with autocast():
                    features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        cls_features, cls_inds, caption_features = None, None, None

        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()
        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))
        
        if self.dynamic_classifier and ann_type != 'caption':
            cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
            ind_with_bg = cls_inds[0].tolist() + [-1]
            cls_features = self.roi_heads.box_predictor[
                0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

        classifier_info = cls_features, cls_inds, caption_features
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)

        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances)
        else:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                ann_type=ann_type, classifier_info=classifier_info)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        if self.with_image_labels:
            if ann_type in ['box', 'prop', 'proptag']:
                losses.update(proposal_losses)
            else: # ignore proposal loss for non-bbox data
                losses.update({k: v * 0 for k, v in proposal_losses.items()})
        else:
            losses.update(proposal_losses)
        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
        
        if self.return_proposal:
            return proposals, losses
        else:
            return losses
        
    
    def preprocess_spatial_memory(self, batched_inputs):
        """
        Preprocess spatial memory
        """
        memory = []
        projection = []
        for x in batched_inputs:
            memory.append(torch.tensor(x['memory'], dtype=torch.half))
            projection.append(torch.tensor(x['proj_indices'], dtype=torch.long).squeeze(2))
        # memory = torch.stack(memory)
        # projection = torch.stack(projection)
        return memory, projection

    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32, 
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
                if has_caption_feature else None # (NB) x (D + 1)
        return caption_features


    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C, 
            weight=freq_weight)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)
        return inds, cls_id_map

@META_ARCH_REGISTRY.register()
class CustomRCNNRecurrent(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        **kwargs):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = True

        # adjust downsample parameter so that it aligns with the dataset used (should be done autonomously)
        # 10, 25
        self.downsample = 10

        # memory update params
        self.map_conditioned = kwargs.pop('map_conditioned')
        self.memory_type = kwargs.pop('memory_type')
        self.save_semmap = kwargs.pop('save_semmap')
        self.output_dir = kwargs.pop('output_path')
        self.cls_score_thresh = kwargs.pop('cls_score_thresh')
        self.obs_score_thresh = kwargs.pop('obs_score_thresh')
        self.test_type = kwargs.pop('test_type')
        self.zs_weight_path = kwargs.pop('zero_shot_weight')
        self.clip_embeddings = np.load(self.zs_weight_path)
        self.zs_weight = torch.tensor(
                self.clip_embeddings, 
                dtype=torch.float32).permute(1, 0).contiguous().cuda() # D x C
        self.zs_weight = torch.cat(
            [self.zs_weight, self.zs_weight.new_zeros((512, 1))], 
            dim=1) # D x (C + 1)
        self.zs_weight = torch.nn.functional.normalize(self.zs_weight, p=2, dim=0)

        # memory gt params
        with open('SMNet/semmap_GT_info.json', 'r') as f:
            self.semmap_gt_info = json.load(f)

        # replica map information
        with open('SMNet/replica_map_info.json', 'r') as f:
            self.replica_map_info = json.load(f)

        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
        })
        ret['map_conditioned'] = cfg.MODEL.TIMM.BASE_NAME == 'resnet50_in21k_map'
        ret['memory_type'] = cfg.MODEL.MEMORY_TYPE
        ret['zero_shot_weight'] = cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH
        ret['save_semmap'] = cfg.MODEL.TEST_SAVE_SEMMAP
        ret['output_path'] = cfg.OUTPUT_DIR
        ret['cls_score_thresh'] = cfg.MODEL.MEMORY_CLS_SCORE_THRESH
        ret['obs_score_thresh'] = cfg.MODEL.MEMORY_OBS_SCORE_THRESH
        ret['test_type'] = cfg.MODEL.TEST_TYPE

        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
        
        return ret

    def forward(self, batched_inputs: List[List[Dict[str, torch.Tensor]]]):
        """
        Sequential pass of model, where the memory is iteratively returned and passed to the model with the next frame
        """
        batch_output = None
        # iterate over the batch of sequences
        for input_seq in batched_inputs:
            # iterate over the sequence
            for i, frame in enumerate(input_seq):

                if self.training:
                    
                    ################################ ADD MEMORY TO INPUT ################################
                    
                    if self.memory_type == 'implicit_memory':
                        frame['memory'], frame['proj_indices'] = self.create_implicit_memory(frame)

                    ################################ INFERENCE ################################

                    # generate output    
                    proposals, output, _ = self.forward_model([frame]) 
                    
                    ################################ OUTPUT UPDATE ################################

                    # merge the output batch
                    if batch_output:
                        # merge each loss term
                        for key in output.keys():
                            batch_output[key] += output[key]
                    else:
                        batch_output = output

                else:
                    
                    ################################ MEMORY RESET ################################

                    if frame['memory_reset']:
                        # for the memory conditioned approach
                        self.semmap_features = None
                        self.observation_count = None
                        # define the data to be saved to file
                        self.semmap = np.full((input_seq[0]['memory'].shape[0],), -1, dtype=np.int32)
                        self.implicit_memory = np.full((input_seq[0]['memory'].shape[0],512), 0, dtype=np.float32)
                        self.observations = np.full((input_seq[0]['memory'].shape[0],), 0, dtype=np.float32)
                    
                    # update the semmap memory at the start of every episode
                    if i == 0:
                        # UPDATE MEMORY AT THE START OF EACH EPISODE FOR THE LONG-TERM MEMORY TEST
                        if self.test_type == 'longterm':
                            updated_memory = self.implicit_memory
                            updated_observations = self.observations
                    
                    # update the memory every step
                    if self.test_type in ['default', 'episodic']:
                        updated_memory = self.implicit_memory
                        updated_observations = self.observations

                    ################################ ADD MEMORY TO INPUT ################################

                    # save the memory and proj_indices
                    memory = torch.from_numpy(frame['memory']).cuda()
                    proj_indices = torch.from_numpy(frame['proj_indices']).cuda().to(torch.long).squeeze(2)

                    # use the real time memory to perform inference
                    frame['memory'] = updated_memory
                    frame['observations'] = updated_observations
                    
                    # use implicit memory embeddings
                    if self.memory_type == 'implicit_memory':
                        frame['memory'], frame['proj_indices'] = self.create_implicit_memory(frame, visualise=False)

                    ################################ INFERENCE ################################

                    # run inference      
                    proposals, output, _ = self.inference([frame])

                    ################################ MEMORY UPDATE ################################

                    # update the implicit memory using the proposals
                    self.update_implicit_memory(proposals, proj_indices, memory, frame, visualise=False)

                    # use the predictions to generate a sem map
                    if self.save_semmap:
                        
                        # save the semmap to file
                        if i==0:
                            # create output dir if needed
                            os.makedirs(os.path.join(self.output_dir, 'semmap'), exist_ok=True)

                            print('Saving memory')
                            # write the memory to h5_file
                            with h5py.File(os.path.join(self.output_dir, 'semmap', frame['sequence_name']), 'w') as f:
                                f.create_dataset('semmap', data=self.semmap, dtype=np.int32)
                                f.create_dataset('impicit_memory', data=self.implicit_memory, dtype=np.float32)
                                f.create_dataset('observations', data=self.observations, dtype=np.float32)

                    ################################ OUTPUT UPDATE ################################

                    # merge the output batch
                    if batch_output:
                        # extend the list of instances
                        batch_output.extend(output)
                    else:
                        batch_output = output
            
        # calculate the average loss
        if self.training:
            for key in batch_output.keys():
                batch_output[key] /= (len(batched_inputs)*len(batched_inputs[0]))

        return batch_output
    
    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)

        # also need to preprocess memory here...
        if self.map_conditioned:
            # get the memory and projection tensors
            memory, projection, observations = self.preprocess_spatial_memory(batched_inputs)
            # memory, projection, observations = batched_inputs[0]['memory'], batched_inputs[0]['proj_indices'], batched_inputs[0]['observations']
            # features = self.backbone(images.tensor, memory, projection, sequence_name=batched_inputs[0]['sequence_name'])
            features, new_memory = self.backbone(images.tensor, memory, projection, observations)
        else:
            features = self.backbone(images.tensor)
        
        proposals, _ = self.proposal_generator(images, features, None)
        results, proposals = self.roi_heads(images, features, proposals)

        # add the segmentations to the proposals
        mask = self.roi_heads.forward_mask_memory(features, proposals)
        mask_rcnn_inference(mask, proposals)

        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return proposals, CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes), new_memory
        else:
            return proposals, results, new_memory

    def forward_model(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        # if not self.training:
        #     return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if self.with_image_labels:
            for inst, x in zip(gt_instances, batched_inputs):
                inst._ann_type = x['ann_type']
                inst._pos_category_ids = x['pos_category_ids']
            ann_types = [x['ann_type'] for x in batched_inputs]
            assert len(set(ann_types)) == 1
            ann_type = ann_types[0]
            if ann_type in ['prop', 'proptag']:
                for t in gt_instances:
                    t.gt_classes *= 0
        
        if self.fp16: # TODO (zhouxy): improve
            if self.map_conditioned:
                with autocast():
                    # get the memory and projection tensors
                    memory, projection, observations = self.preprocess_spatial_memory(batched_inputs)
                    # create memory tensor and projection tensor
                    features, new_memory = self.backbone(images.tensor.half(), memory, projection, observations)
            else:
                with autocast():
                    features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        cls_features, cls_inds, caption_features = None, None, None

        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()
        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))
        
        if self.dynamic_classifier and ann_type != 'caption':
            cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
            ind_with_bg = cls_inds[0].tolist() + [-1]
            cls_features = self.roi_heads.box_predictor[
                0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

        classifier_info = cls_features, cls_inds, caption_features
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        

        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances)
        else:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                ann_type=ann_type, classifier_info=classifier_info)
        
        # add the segmentations to the proposals
        seg = self.roi_heads.forward_mask_memory(features, proposals)
        mask_rcnn_inference(seg, proposals)
            
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        if self.with_image_labels:
            if ann_type in ['box', 'prop', 'proptag']:
                losses.update(proposal_losses)
            else: # ignore proposal loss for non-bbox data
                losses.update({k: v * 0 for k, v in proposal_losses.items()})
        else:
            losses.update(proposal_losses)
        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
        
        if self.return_proposal:
            return proposals, losses, new_memory
        else:
            return losses, new_memory
        
    def update_implicit_memory(self, proposals, proj_indices, memory, frame, visualise=False):
        # get the inference results from the proposals and features
        inference_results = self.inference_with_proposals(proposals, self.zs_weight, self.cls_score_thresh)
        
        # if there are detected objects
        if inference_results is not None:
            # extract the results
            boxes, box_features, masks, pred_instances = inference_results
            # project the box features into the image frame
            image_features, observed_pixels = self.box_to_image_features(box_features, masks)

            # get the the projected features
            proj_features, observed_mem = self.project_image_features(image_features, observed_pixels, [proj_indices], [memory])
            
            # convert these features to the 2D map
            semmap_update = torch.zeros(memory.shape[0], 512, dtype=torch.float).cuda()
            semmap_update[observed_mem] = proj_features

            unique_indices = torch.unique(proj_indices)
            observed_update = torch.zeros(memory.shape[0], 1, dtype=torch.float).cuda()
            observed_update[unique_indices] = 1
           
            # get the dimensions of the map
            try:
                map_w, _, map_h = self.semmap_gt_info[frame['sequence_name'][0:13]]['dim']
                map_h = math.ceil(map_h / self.downsample)
                map_w = math.ceil(map_w / self.downsample)
            except KeyError:
                split_name = frame['sequence_name'].split('_')
                if len(split_name) == 5:
                    house = split_name[0] + '_' + split_name[1] + '_' + split_name[2]
                    level = split_name[3]
                elif len(split_name) == 4:
                    house = split_name[0] + '_' + split_name[1]
                    level = split_name[2]
                else:
                    house = split_name[0] + '_' + split_name[1]
                    level = None
                
                if level is not None:
                    env = '_'.join((house, level))
                else:
                    env = house

                try:
                    map_w, _, map_h = self.replica_map_info[env]['dim']

                # default to using 200x200
                except KeyError:
                    map_w, map_h = 200, 200

            semmap_update = semmap_update.reshape(map_h, map_w, 512)
            semmap_update = semmap_update.permute(2, 0, 1).unsqueeze(0)

            observed_update = observed_update.reshape(map_h, map_w, 1)
            observed_update = observed_update.permute(2, 0, 1)

            # update the map
            if self.semmap_features is None:
                self.semmap_features = semmap_update
                self.observation_count = observed_update
            else:
                # self.semmap_features = (semmap*i + semmap_update)/(i+1)
                self.semmap_features = self.semmap_features + semmap_update
                self.observation_count = self.observation_count + observed_update
            
            # mask the memory based on intensity of observations
            # calculate the average feature intensity across the second dimension
            observation_intensity = torch.mean(self.semmap_features.abs(), dim=1)
            observation_intensity[self.observation_count > 1] = observation_intensity[self.observation_count > 1]/self.observation_count[self.observation_count > 1]

            # normalise to between 0 and 1
            observation_intensity = (observation_intensity - torch.min(observation_intensity))/(torch.max(observation_intensity) - torch.min(observation_intensity))
            # Display the grayscale image
            if visualise:
                cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Grayscale Image", (255*observation_intensity.permute(1,2,0)).cpu().numpy().astype(np.uint8))
            
            self.semmap = self.visualise_clip_image_features(self.semmap_features, self.zs_weight, 'semmap', mask = observation_intensity, thresh=self.obs_score_thresh, visualise=visualise)

            # convert the features and observations to numpy to save to file
            self.implicit_memory = self.semmap_features.squeeze().permute(1,2,0).reshape(-1, 512)
            self.implicit_memory = self.implicit_memory.cpu().numpy()
            self.observations = self.observation_count.squeeze().reshape(-1)
            self.observations = self.observations.cpu().numpy()

    def create_implicit_memory(self, frame, visualise=False):
        # get the current memory features
        memory_features = torch.from_numpy(frame['memory']).cuda()
        observations = torch.from_numpy(frame['observations']).cuda()
        proj_indices = frame['proj_indices']

        # scale features based on intensity of observation
        memory_features[observations > 1] = memory_features[observations > 1] / (observations.unsqueeze(1)[observations > 1])

        # # update memory
        memory = memory_features.cpu().numpy()
        
        # visualise the results
        if visualise:
            # get the dimensions of the map
            try:
                map_w, _, map_h = self.semmap_gt_info[frame['sequence_name'][0:13]]['dim']
                map_h = math.ceil(map_h / self.downsample)
                map_w = math.ceil(map_w / self.downsample)
            
            except KeyError:
                split_name = frame['sequence_name'].split('_')
                if len(split_name) == 5:
                    house = split_name[0] + '_' + split_name[1] + '_' + split_name[2]
                    level = split_name[3]
                elif len(split_name) == 4:
                    house = split_name[0] + '_' + split_name[1]
                    level = split_name[2]
                else:
                    house = split_name[0] + '_' + split_name[1]
                    level = None
                
                if level is not None:
                    env = '_'.join((house, level))
                else:
                    env = house

                map_w, _, map_h = self.replica_map_info[env]['dim']

            # the magnitude of the features defines the intensity of the observation
            observation_intensity = torch.mean(memory_features.abs(), dim=1)
            # normalise to between 0 and 1
            if torch.max(observation_intensity) > 0:
                observation_intensity = (observation_intensity - torch.min(observation_intensity))/(torch.max(observation_intensity) - torch.min(observation_intensity))
            # reshape
            observation_intensity = observation_intensity.reshape(map_h, map_w, 1)
            # Display the grayscale image
            # cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
            # cv2.imshow("Grayscale Image", (255*observation_intensity).cpu().numpy().astype(np.uint8))

            # the region embeddings can be dot producted to created an explicit map
            memory_features = memory_features.reshape(map_h, map_w, 512)
            memory_features = memory_features.permute(2, 0, 1).unsqueeze(0)
            # visualise the update
            semmap = self.visualise_clip_image_features(memory_features, self.zs_weight, 'semmap', mask = observation_intensity, thresh=0.4, visualise=visualise)

        return memory, proj_indices
    
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
    
    def project_image_features(self, image_features, observed_pixels, proj_indices, memory):
        # extract only those image features and projection indices that have been observed, and reshape to flat tensors
        image_features = image_features[:,:,observed_pixels].squeeze(0)
        image_features = image_features.permute(1,0).reshape(-1, 512)
        proj = proj_indices[0][observed_pixels]

        # project the image features into a new space using 1x1 layer
        # image_features = self.map_merge_memory_projection(image_features).squeeze(0)
        
        # self.downsample each tensor to be the same size
        proj = proj[::8]
        image_features = image_features[::8]

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

        return mean, observed_mem
    
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

    def preprocess_spatial_memory(self, batched_inputs):
        """
        Preprocess spatial memory
        """
        memory = []
        projection = []
        observations = []   
        for x in batched_inputs:
            # convert memory and proj indices to tensor
            if not torch.is_tensor(x['memory']):
                x['memory'] = torch.from_numpy(x['memory']).cuda()
            if not torch.is_tensor(x['proj_indices']):
                x['proj_indices'] = torch.from_numpy(x['proj_indices']).cuda().squeeze(2)
            if not torch.is_tensor(x['observations']) and x['observations'] is not None:
                x['observations'] = torch.from_numpy(x['observations']).cuda().to(torch.half)

            # add to list of tensors
            memory.append(x['memory'].to(torch.half))
            projection.append(x['proj_indices'].to(torch.long))
            observations.append(x['observations'])

        # memory = torch.stack(memory)
        # projection = torch.stack(projection)
        return memory, projection, observations

    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32, 
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
                if has_caption_feature else None # (NB) x (D + 1)
        return caption_features


    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C, 
            weight=freq_weight)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)
        return inds, cls_id_map