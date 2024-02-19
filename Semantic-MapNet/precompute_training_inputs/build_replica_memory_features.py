import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from SMNet.model_test import SMNet
from projector import _transform3D
from projector.projector import Projector
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 

from utils.habitat_utils import HabitatUtils
from utils import convert_weights_cuda_cpu

from semseg.rednet import RedNet
import h5py
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from utils.crop_memories import crop_memories

from torch_scatter import scatter_add
from tqdm import tqdm
import math

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

# load data/env_splits.json
with open('data/envs_splits.json', 'r') as f:
    env_splits = json.load(f)

#Settings
resolution = 0.02 # topdown resolution
res_downsample = 10 # downsample factor - 10, 25
default_ego_dim = (480, 640) #egocentric resolution
z_clip = 0.50 # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0

# now update the training data with the memory
# semmap_dir = 'data/semmap/'

# data and output dir for training
data_dir = 'data/training/replica_data'
out_dir = 'data/training/replica_memory_res{}'.format(resolution*res_downsample)

# data and output dir for testing
# data_dir = 'data/test_data/smnet_test_data'
# out_dir = 'data/test_data/implicit_memory_res{}'.format(resolution*res_downsample)

# semmap_info = json.load(open('data/semmap_GT_info.json', 'r'))
map_info = json.load(open('data/training/replica_map_info.json', 'r'))

# create outdir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

files = os.listdir(data_dir)

# memory_files = os.listdir(memory_dir)
# memory_files = [x for x in memory_files if 'h5' in x]
# memory_files = [x for x in memory_files if len(x)==16]
# for n, mem_file in tqdm(enumerate(memory_files), total=len(memory_files)):

#     house, level = mem_file[0:13].split('_')
#     print(house, level)
#     env = '_'.join((house, level))

#     # get details of the scene
#     map_world_shift = np.array(semmap_info[env]['map_world_shift'])
#     world_discret_dim = np.array(semmap_info[env]['dim'])
#     map_width = world_discret_dim[0]
#     map_height = world_discret_dim[2]
#     print(map_height, map_width)

#     # get ground truth semantic map
#     h5file = h5py.File(os.path.join('data/semmap/', mem_file[0:13]) + '.h5', 'r')
#     semmap_gt = np.array(h5file['map_semantic'])
#     h5file.close()

#     house_paths = [x for x in files if env in x]
    
#     h5file = h5py.File(os.path.join(memory_dir, mem_file[0:13]) + '.h5', 'r')
#     memory = np.array(h5file['memory'])
#     memory = np.transpose(memory[0], (1, 2, 0))
#     semmap = np.array(h5file['pred_map'])
#     h5file.close()

    # for m, file in tqdm(enumerate(house_paths), total=len(house_paths)):

map_info
for n, file in tqdm(enumerate(files), total=len(files)):
    split_name = file.split('_')
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

    # print(env)
    # print(map_info[env])

    # get details of the scene
    # map_world_shift = np.array(semmap_info[env]['map_world_shift'])
    # world_discret_dim = np.array(semmap_info[env]['dim'])
    map_world_shift = np.array([map_info[env]['x_min']-1, 0, map_info[env]['y_min']-1])
    world_discret_dim = np.array([(2+map_info[env]['x_max']-map_info[env]['x_min'])/resolution, 0, (2+map_info[env]['y_max']-map_info[env]['y_min'])/resolution])
    map_width = world_discret_dim[0]
    map_height = world_discret_dim[2]

    # get ground truth semantic map
    # h5file = h5py.File(os.path.join('data/semmap/', file[0:13]) + '.h5', 'r')
    # semmap_gt = np.array(h5file['map_semantic'])
    # h5file.close()

    # downsample the semantic map and memory
    # semmap_gt = semmap_gt[::res_downsample, ::res_downsample]

    # get the height and width of the semantic map after downsampling
    map_height = math.ceil(map_height / res_downsample)
    map_width = math.ceil(map_width / res_downsample)

    # try:
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

    # flatten the semmap_gt
    # semmap_gt = semmap_gt.reshape(-1)
    
    # convert pixels in map to contain the index of the flattened semmap_gt
    # set maximum width and height in pixels in map
    pixels_in_map[:,:,:,1] = np.clip(pixels_in_map[:,:,:,1], 0, map_height-1)
    pixels_in_map[:,:,:,0] = np.clip(pixels_in_map[:,:,:,0], 0, map_width-1)
    pixels_in_map = pixels_in_map[:,:,:,1] * map_width + pixels_in_map[:,:,:,0]
    pixels_in_map = pixels_in_map[:, :, :,np.newaxis]

    # memory and semmap should be empty
    memory = np.zeros((map_width*map_height, 256))
    semmap = np.zeros((map_width*map_height))

    # # display the projection information
    # pixel_pos = None
    # for i in range(20):

    #     memory_copy = np.zeros((map_width*map_height), dtype=np.int32)
    #     proj_i = pixels_in_map[i,:,:,:]

    #     # get the position of the middle pixel
    #     if pixel_pos is None:
    #         pixel_pos = proj_i[240, 320][0]

    #     img_pixels = []
    #     for k in range(proj_i.shape[0]):
    #         for j in range(proj_i.shape[1]):
    #             proj = proj_i[k,j][0]
    #             if proj == pixel_pos:
    #                 memory_copy[proj] = -2
    #                 img_pixels.append([k,j])
    #             else:
    #                 memory_copy[proj] = -1
        
    #     # reshape the semmap_gt_copy to w, h
    #     memory_print = memory_copy.reshape(map_height, map_width)

    #     # replace semmap_gt_copy with the colour in pallete
    #     memory_print = palette[memory_print]

    
    #     plt.figure(0)
    #     plt.imshow(memory_print)
    #     plt.title('Topdown semantic map prediction')
    #     plt.axis('off')
    #     plt.show(block=False)

    #     # plot in cv2
    #     img = rgb[i]
    #     for k in range(len(img_pixels)):
    #         img[img_pixels[k][0], img_pixels[k][1]] = [255, 0, 0]
    #     cv2.imshow('RGB', img)
    #     cv2.waitKey(0)

    #     plt.close()

    # print(memory.shape)
    # print(pixels_in_map.shape)
    # print(semmap.shape)

    map_info[env]["dim"] = [map_width, 0, map_height]

    # write to NEW numpy file
    with h5py.File(os.path.join(out_dir, file), 'w') as f:
        f.create_dataset('memory_features', data=memory, dtype=np.float32)
        f.create_dataset('proj_indices', data=pixels_in_map, dtype=np.int32)
        f.create_dataset('semmap_gt', data=semmap, dtype=np.int32)

    # except Exception as e:
    #     print(e)
    #     pass

json.dump(map_info, open('data/training/replica_map_info.json', 'w'))