import os
import h5py
import json
import torch
import numpy as np
import torch.nn.functional as F
import ast
from torch.utils import data
import math

import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import time
from PIL import Image
from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
import cv2

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



def collate_smnet(batch):
    # print(batch)

    return batch
    
class SMNetDetectionLoader(data.Dataset):
    def __init__(self, data_path = 'sensor_data', test_type = 'default', clip_path = None, memory_type = '', semmap_path = 'gt'):

        self.clip_path = clip_path
        self.memory_type = memory_type
        # implicit_memory_res0.1, smnet_training_data_memory_2, implicit_memory_noise0.1_res0.2, replica_memory_res0.2
        self.memory_path = os.path.join(data_path, 'memory_data')
        self.data_path = os.path.join(data_path, 'sensor_data')
        self.image_root = os.path.join(data_path, 'JPEGImages')
        print(self.memory_path)
        print(self.data_path)
        self.test_type = test_type
        # can reduce to lower GPU memory
        self.max_sequence_length = 20

        # choose which semmap to use - default of False will use the gt
        self.semmap_path = semmap_path
        print(self.semmap_path)

        # map gt params
        with open('SMNet/semmap_GT_info.json', 'r') as f:
            self.semmap_gt_info = json.load(f)

        # if split in ['train', 'val', 'test']:
        #     self.root += 'training/'
        #     self.data_path = 'smnet_training_data_2'
        # elif 'replica' in split:
        #     self.root += 'training/'
        #     self.data_path = 'replica_data'
        #     self.memory_path = 'replica_memory_res0.2'
        # elif split == 'test':
        #     self.root += 'test_data/'
        #     self.data_path = 'smnet_test_data'

        self.files = os.listdir(self.memory_path)

        # sort the files
        def custom_sort(string):
            string_split = string.split('_')
            name = ''
            for i in range(len(string_split)-1):
                name += string_split[i] + '_'
            # name = string.split('_')[0] + '_' + string.split('_')[1]
            num = int(string.split('_')[-1].split('.')[0])
            return (name, num)
        self.files = sorted(self.files, key=custom_sort)

        # extend the sequence for asessing longer term benefit
        if self.test_type == 'longterm':
            # split the list into chunks of 50
            self.files = [self.files[i:i + 50] for i in range(0, len(self.files), 50)]
            self.files = self.files * 2
            self.files = sorted(self.files)
            # flatten the files
            self.files = [item for sublist in self.files for item in sublist]
            # 
            for j in range(50, len(self.files), 100):
                self.files[j] = self.files[j-1]

        self.envs = [x.split('.')[0] for x in self.files]

        # -- load semantic map GT
        # h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_semmap.h5'), 'r')
        # self.semmap_GT = np.array(h5file['semantic_maps'])
        # h5file.close()
        # self.semmap_GT_envs =json.load(open(os.path.join(self.root,'smnet_training_data_semmap.json'), 'r'))
        # self.semmap_GT_indx = {i: self.semmap_GT_envs.index(self.envs[i] + '.h5') for i in range(len(self.files))}
        
        # info for map projection
        self.semmap_info = json.load(open('SMNet/semmap_GT_info.json', 'r'))
        self.resolution = 0.02

        # select which classes are to be included for detections
        # object_lvis = ['bed', 'stool', 'towel', 'fireplace', 'picture', 'cabinet', 'toilet', 'curtain', 'lighting', 'table', 'shelving', 'mirror', 'sofa', 'cushion', 'bathtub', 'chair', 'chest_of_drawers', 'sink', 'seating', 'tv_monitor']
        # reduced_object_lvis = ['bed', 'towel', 'fireplace', 'picture', 'cabinet', 'toilet', 'curtain', 'table', 'sofa', 'cushion', 'bathtub', 'chair', 'chest_of_drawers', 'sink', 'tv_monitor']
        self.class_ids = [0, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 19]
        
        # mapping from object_lvis to smnet for semmap gt
        self.smnet_class_mapping = [0, 11, 17, 1, 14, 4, 13, 10, 16, 6, 0, 0, 18]

        # if we are running the semantic_gt baseline, load the clip embeddings
        if self.clip_path:
            # np.load to load the clip embeddings
            self.clip_embeddings = np.load(self.clip_path)

        # # -- load projection indices
        # if self.ego_downsample:
        #     h5file=h5py.File(os.path.join(self.root,'smnet_training_data_maxHIndices_every4_{}.h5'.format(split)), 'r')
        #     self.projection_indices = np.array(h5file['indices'])
        #     self.masks_outliers = np.array(h5file['masks_outliers'])
        #     h5file.close()
        #     self.projection_indices_envs = \
        #     json.load(open(os.path.join(self.root,'smnet_training_data_maxHIndices_every4_{}.json'.format(split)), 'r'))
        # else:
        #     print('loading projection indices')
        #     h5file=h5py.File(os.path.join(self.root,'smnet_training_data_maxHIndices_{}.h5'.format(split)), 'r')
        #     self.projection_indices = np.array(h5file['indices'])
        #     self.masks_outliers = np.array(h5file['masks_outliers'])
        #     h5file.close()
        #     self.projection_indices_envs = \
        #     json.load(open(os.path.join(self.root,'smnet_training_data_maxHIndices_{}.json'.format(split)), 'r'))

        # self.projection_indices_indx = {i:self.projection_indices_envs.index(self.envs[i]) for i in range(len(self.files))}

        assert len(self.files) > 0

        self.available_idx = list(range(len(self.files)))

    def __len__(self):
        return len(self.available_idx)


    def __getitem__(self, index):
        start = time.time()
        env_index = self.available_idx[index]

        file = self.files[env_index]
        env  = self.envs[env_index]

        # we want to build up a series of detection batches
        detection_batch = []

        # from a series of episodes
        episodes = [file]

        # if we are training, we select a second file from the same environment
        # if self.split == 'train':
        #     # get the name of the current scene
        #     scene = file.split('_')[0] + '_' + file.split('_')[1]

        #     # get the index of all the files with the same scene name
        #     scene_idx = [i for i in range(len(self.files)) if scene in self.files[i]]

        #     # randomly select a second file from the same scene
        #     file2 = self.files[np.random.choice(scene_idx)]

        #     # add to episodes
        #     episodes.append(file2)
        
        # iterate through each episode
        for file in episodes:

            # get the spatial memory for the current environment
            try:
                h5file = h5py.File(os.path.join(self.memory_path, file), 'r')
                memory = np.array(h5file['memory_features'])
                semmap_gt = np.array(h5file['semmap_gt'])
                proj_indices = np.array(h5file['proj_indices'])
                h5file.close()
            except Exception as e:
                print(e)
                memory = np.zeros((1,256))
                proj_indices = np.zeros((20, 480, 640, 1))

            # import the realtime semmap
            # check if self.semmap_path is a path to a directory

            if os.path.exists(self.semmap_path):
                h5file = h5py.File(os.path.join(self.semmap_path, file), 'r')
                semmap_real = np.array(h5file['semmap'])
                implicit_memory = np.array(h5file['impicit_memory'])
                observations = np.array(h5file['observations'])
                # adjust the semmap indices up by 1 (empty space needs to change from -1 to 0)
                semmap_real = semmap_real + 1
                h5file.close()
            else:
                semmap_real = None
                implicit_memory = memory
                observations = None

                # display the generated semmap
                # self.display_semmap(semmap_real-1, file)

            # if we are running the semantic_gt or map_gt baseline, load the clip embeddings
            if self.clip_path:
                memory = self.clip_embeddings
                # insert a row of zeros
                memory = np.insert(memory, 0, np.zeros((1,512)), axis=0)
                # if we are running the semnatic_map_gt, we need to map the indices
                if self.memory_type == 'map_gt':
                    if semmap_real is not None:
                        # we are using our generated embeddings
                        print('using generated semmap')
                        proj_indices = semmap_real[proj_indices]
                    else:
                        # when using the semantic map ground truth 
                        memory = memory[self.smnet_class_mapping]
                        proj_indices = semmap_gt[proj_indices]

            h5file = h5py.File(os.path.join(self.data_path, file), 'r')

            # build the object detection data required to train detic models
            rgb = np.array(h5file['rgb'])
            segmentation_data = np.array(h5file['segmentation_data'])

            # for i in range(len(h5file['detection_data'])):
            for i in range(min(self.max_sequence_length, len(h5file['detection_data']))):
                # decode detection data
                detection_data = str(h5file['detection_data'][i].decode())
                detection_data = detection_data.replace("'", "\"")
                file_name = detection_data.split('"file_name": ')[1].split(', "image": ')[0]
                gt_box, gt_class = detection_data.split('"gt_boxes": ')[1].split(', "gt_classes": ')
                gt_class = ast.literal_eval(gt_class[:-1])
                gt_box = ast.literal_eval(gt_box)
                gt_box = [[gt_box[i][0], gt_box[i][1], gt_box[i][2]+gt_box[i][0], gt_box[i][3]+gt_box[i][1]] for i in range(len(gt_box))]

                # filter out classes that are not in the class_ids list
                gt_box = [gt_box[i] for i in range(len(gt_box)) if gt_class[i] in self.class_ids]
                gt_class = [gt_class[i] for i in range(len(gt_class)) if gt_class[i] in self.class_ids]  

                # if the memory is set as semantic_gt, ready it for the dataloader
                if self.clip_path:
                    # the proj_indeces are simply the segmentation data
                    if self.memory_type == 'semantic_gt':
                        proj_indices[i] = segmentation_data[i].reshape(segmentation_data.shape[1], segmentation_data.shape[2], 1)
                    # else:
                    #     print(proj_indices[i].shape)
                    #     print(semmap_gt.shape)
                    #     proj_indices[i] = semmap_gt[proj_indices[i]]
                
                # manually import the image, same as done in original DETIC dataloader
                with PathManager.open(os.path.join(self.image_root, file_name[1:-1]), "rb") as f:
                    rgb_i = Image.open(f)
                    # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
                    rgb_i = _apply_exif_orientation(rgb_i)
                    rgb_i = convert_PIL_to_numpy(rgb_i, 'RGB')
                # except FileNotFoundError:
                #     rgb_i = rgb[i]

                # define memory update
                if self.test_type in ['default', 'longterm']:
                    seq_id = int(file.split('_')[-1].split('.')[0])
                    mem_reset = (seq_id == 0 and i==0)
                elif self.test_type == 'episodic':
                    mem_reset = (i==0)

                # return values
                detection_batch.append({'file_name': file_name[1:-1], 'sequence_name': file, 'gt_boxes': np.array(gt_box), 'gt_classes': np.array(gt_class), 'image': rgb_i, 'proj_indices': proj_indices[i], 'memory_reset': mem_reset})
                # append memory
                if self.memory_type in ['explicit_map', 'implicit_memory']:
                    detection_batch[-1]['memory_features'] = implicit_memory
                    detection_batch[-1]['observations'] = observations
                else:
                    detection_batch[-1]['memory_features'] = memory
                    detection_batch[-1]['observations'] = None

        # close the h5file
        h5file.close()

        return detection_batch


