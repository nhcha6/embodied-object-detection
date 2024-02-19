import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

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
      [220, 220, 220]])

# load data/env_splits.json
with open('data/envs_splits.json', 'r') as f:
    env_splits = json.load(f)

# now update the training data with the memory
semmap_dir = 'data/semmap/'
data_dir = 'data/training/smnet_training_data_2'
memory_dir = 'data/training/smnet_training_data_memory'
out_dir = 'data/training/smnet_training_data_memory_2'
semmap_info = json.load(open('data/semmap_GT_info.json', 'r'))

# create outdir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#Settings
resolution = 0.02 # topdown resolution
default_ego_dim = (480, 640) #egocentric resolution
z_clip = 0.50 # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0

files = os.listdir(data_dir)

memory_files = os.listdir(memory_dir)
memory_files = [x for x in memory_files if 'h5' in x]
memory_files = [x for x in memory_files if len(x)==16]
for n, mem_file in tqdm(enumerate(memory_files), total=len(memory_files)):

    house, level = mem_file[0:13].split('_')
    print(house, level)
    env = '_'.join((house, level))

    # get details of the scene
    map_world_shift = np.array(semmap_info[env]['map_world_shift'])
    world_discret_dim = np.array(semmap_info[env]['dim'])
    map_width = world_discret_dim[0]
    map_height = world_discret_dim[2]
    print(map_height, map_width)

    # get ground truth semantic map
    h5file = h5py.File(os.path.join('data/semmap/', mem_file[0:13]) + '.h5', 'r')
    semmap_gt = np.array(h5file['map_semantic'])
    h5file.close()

    house_paths = [x for x in files if env in x]
    
    h5file = h5py.File(os.path.join(memory_dir, mem_file[0:13]) + '.h5', 'r')
    memory = np.array(h5file['memory'])
    memory = np.transpose(memory[0], (1, 2, 0))
    semmap = np.array(h5file['pred_map'])
    h5file.close()

    for m, file in tqdm(enumerate(house_paths), total=len(house_paths)):
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
            pixels_in_map = (projection_indices[:,:,:, [0,2]] / resolution).round().long()

            # get a list of all grid indices that are viewed in this sequence
            pixels_in_map = pixels_in_map.numpy()
            set_of_indices = set([tuple(x) for x in pixels_in_map.reshape(-1, 2)])
            set_of_indices = np.array([list(x) for x in set_of_indices])
            # extract a flat list of features for these indices
            index_mapping = {tuple(x): i for i, x in enumerate(set_of_indices)}
            set_of_features = memory[set_of_indices[:,1], set_of_indices[:,0], :]
            set_of_classes = semmap_gt[set_of_indices[:,1], set_of_indices[:,0]]

            # update the pixels_in_map to store the new indeces
            s, w, h, _ = pixels_in_map.shape
            pixels_in_map = pixels_in_map.reshape(-1, 2)
            pixels_in_map = np.array([index_mapping[tuple(x)] for x in pixels_in_map])
            pixels_in_map = pixels_in_map.reshape(s, w, h, 1)

            # # check correspondance between new indices and old grid cells
            # # for i in range(pixels_in_map.shape[0]):
            # i=0
            # proj_indices_i = pixels_in_map[i,:,:,:]
            
            # #check to see that the projection indices are aligning with the semantic map and the images
            # semmap_new = semmap.copy()
            # for k in range(proj_indices_i.shape[0]):
            #     for j in range(proj_indices_i.shape[1]):
            #         proj = set_of_indices[proj_indices_i[k,j]][0]
            #         semmap_new[min(proj[1],map_height-1), min(proj[0], map_width-1)] = (0,255,0)

            # # check the egocentric image
            # image_new = rgb[i].copy()
            # for k in range(proj_indices_i.shape[0]):
            #     for j in range(proj_indices_i.shape[1]):
            #         if set_of_classes[proj_indices_i[k,j]]:
            #             image_new[k, j] = palette[set_of_classes[proj_indices_i[k,j]]]
  
            # plt.figure(0)
            # plt.imshow(rgb[i])
            # plt.figure(1)
            # plt.imshow(image_new)
            # plt.figure(2)
            # plt.imshow(semmap_new)
            # plt.title('Topdown semantic map prediction')
            # plt.axis('off')
            # plt.show()

            # if m==2:
                # break

            # write to NEW numpy file
            with h5py.File(os.path.join(out_dir, file), 'w') as f:
                f.create_dataset('memory_features', data=set_of_features, dtype=np.float32)
                f.create_dataset('proj_indices', data=pixels_in_map, dtype=np.int32)
                f.create_dataset('semmap_gt', data=set_of_classes, dtype=np.int32)

            # # check load time
            # start = time.time()
            # h5file = h5py.File(os.path.join(memory_dir, file), 'r')
            # memory_features = np.array(h5file['memory_features'])
            # pixels_in_map = np.array(h5file['proj_indices'])
            # h5file.close()
            # print('Load time: ', time.time() - start)
        except Exception as e:
            print(e)
            pass
