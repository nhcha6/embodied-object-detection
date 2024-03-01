import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from projector import _transform3D
from projector.projector import Projector
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 

from utils.habitat_utils import HabitatUtils
from utils import convert_weights_cuda_cpu

import h5py
import sys
import time

from utils.crop_memories import crop_memories

from torch_scatter import scatter_add
from tqdm import tqdm
import math

import argparse

# define the colour pallette
palette = np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 0, 0],
      ])

if __name__ == '__main__':

    # get args
    parser = argparse.ArgumentParser(description='Save habitat data')
    parser.add_argument('--data_path', type=str, default='./', help='path to Matterport data')
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    for split in splits:

        # if split!='val':
            # continue
    
        # load data/env_splits.json
        with open('./envs_splits.json', 'r') as f:
            env_splits = json.load(f)

        #Settings
        resolution = 0.02 # topdown resolution
        res_downsample = 10 # downsample factor - 10, 25
        default_ego_dim = (480, 640) #egocentric resolution
        z_clip = 0.50 # detections over z_clip will be ignored
        vfov = 67.5
        vfov = vfov * np.pi / 180.0

        # data and output dir for training
        data_dir = f'../embodied_data/mp3d_{split}/sensor_data/'
        # out_dir = 'data/training/implicit_memory_res{}'.format(resolution*res_downsample)
        out_dir = f'../embodied_data/mp3d_{split}/memory_data/'

        semmap_info = json.load(open('./semmap_GT_info.json', 'r'))

        # create outdir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        files = os.listdir(data_dir)
        for n, file in tqdm(enumerate(files), total=len(files)):
            house, level = file[0:13].split('_')
            env = '_'.join((house, level))

            # get details of the scene
            map_world_shift = np.array(semmap_info[env]['map_world_shift'])
            world_discret_dim = np.array(semmap_info[env]['dim'])
            map_width = world_discret_dim[0]
            map_height = world_discret_dim[2]

            # downsample the semantic map and memory
            map_height = math.ceil(map_height / res_downsample)
            map_width = math.ceil(map_width / res_downsample)

            # get ground truth semantic map
            semmap_gt = np.zeros((map_width*map_height))

            try:
                # load the projection information
                h5file = h5py.File(os.path.join(data_dir, file), 'r')
                projection_indices = np.array(h5file['projection_indices'], dtype=np.float32)
                masks_outliers = np.array(h5file['masks_outliers'], dtype=np.bool)
                sensor_positions = np.array(h5file['sensor_positions'], dtype=np.float32)
                # rgb for visualisation
                rgb = np.array(h5file['rgb'])
                h5file.close()

                # transfer to Pytorch
                projection_indices = torch.FloatTensor(projection_indices)
                masks_outliers = torch.BoolTensor(masks_outliers)
                sensor_positions = torch.FloatTensor(sensor_positions)
                map_world_shift = torch.FloatTensor(map_world_shift)
                projection_indices -= map_world_shift
                pixels_in_map = (projection_indices[:,:,:, [0,2]] / (resolution*res_downsample)).round().long()
                pixels_in_map = pixels_in_map.numpy()

                # convert pixels in map to contain the index of the flattened semmap_gt
                # set maximum width and height in pixels in map
                pixels_in_map[:,:,:,1] = np.clip(pixels_in_map[:,:,:,1], 0, map_height-1)
                pixels_in_map[:,:,:,0] = np.clip(pixels_in_map[:,:,:,0], 0, map_width-1)
                pixels_in_map = pixels_in_map[:,:,:,1] * map_width + pixels_in_map[:,:,:,0]
                pixels_in_map = pixels_in_map[:, :, :,np.newaxis]

                # memory should be empty to start with
                memory = np.zeros((semmap_gt.shape[0], 256))

                # write to NEW numpy file
                with h5py.File(os.path.join(out_dir, file), 'w') as f:
                    f.create_dataset('memory_features', data=memory, dtype=np.float32)
                    f.create_dataset('proj_indices', data=pixels_in_map, dtype=np.int32)
                    f.create_dataset('semmap_gt', data=semmap_gt, dtype=np.int32)

            except Exception as e:
                print(e)
                pass
