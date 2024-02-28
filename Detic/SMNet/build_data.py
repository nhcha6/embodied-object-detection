import os
import sys
import json
import h5py
import torch
import numpy as np
from tqdm import tqdm
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from projector import _transform3D
from projector.point_cloud import PointCloud

from utils.habitat_utils import HabitatUtils
from utils import convert_weights_cuda_cpu

from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms

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
      [220, 220, 220]])

if __name__ == '__main__':

    # get args
    parser = argparse.ArgumentParser(description='Save habitat data')
    parser.add_argument('--data_path', type=str, default='./', help='path to Matterport data')
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    for split in splits:

        if split != 'val':
            continue 
    
        # -- settings
        # output_dir = 'data/training/smnet_training_data_2/'
        output_dir = f'../embodied_data/mp3d_{split}/sensor_data/'
        os.makedirs(output_dir, exist_ok=True)

        #Settings
        resolution = 0.02 # topdown resolution
        default_ego_dim = (480, 640) #egocentric resolution
        z_clip = 0.50 # detections over z_clip will be ignored
        vfov = 67.5
        vfov = vfov * np.pi / 180.0
        features_spatial_dimensions = (480,640)


        nb_samples_per_env = 50
        nb_frames_per_sample = 20

        paths = json.load(open('./paths.json', 'r'))

        device = torch.device('cuda')

        normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

        depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

        # -- build projector
        map_world_shift = np.zeros(3)
        world_shift_origin=torch.from_numpy(map_world_shift).float().to(device=device)
        projector = PointCloud(vfov, 1,
                            features_spatial_dimensions[0],
                            features_spatial_dimensions[1],
                            world_shift_origin,
                            z_clip,
                            device=device)

        # open env_psplits
        with open('./envs_splits.json', 'r') as f:
            env_splits = json.load(f)
        envs = env_splits[split+ '_envs']
        
        # open data info
        with open('./info_mp3d_data.json', 'r') as f:
            info = json.load(f)

        """
        -->> START
        """
        # for env, path in tqdm(paths.items()):
        for env in tqdm(envs):
            try:
                path = paths[env]
            except KeyError:
                continue

            # if env in ['YFuZgdQ5vWj_0', 'X7HyMhZNoso_0', 'Vvot9Ly1tCj_0']:
            #     pass
            # else:
            #     continue
        
            house, level = env.split('_')

            # scene = '../Matterport/habitat_data/v1/tasks/mp3d/{}/{}.glb'.format(house, house)
            scene = os.path.join(args.data_path, 'habitat_data/v1/tasks/mp3d/{}/{}.glb'.format(house, house))
            habitat = HabitatUtils(scene, int(level))

            N = len(path['positions'])

            

            # for m in range(nb_samples_per_env):

            #     start = np.random.randint(0, high=N-nb_frames_per_sample)

            #     info[env][m] = {'start':start}
        
            for m, seq in info[env].items():

                filename = os.path.join(output_dir, env+'_{}.h5'.format(m))
                
                start = seq['start']

                sub_path = {}
                sub_path['positions'] = path['positions'][start:start+nb_frames_per_sample+1]
                sub_path['orientations'] = path['orientations'][start:start+nb_frames_per_sample+1]
                sub_path['actions'] = path['actions'][start:start+nb_frames_per_sample+1]

                frames_RGB = []
                frames_depth = []
                sensor_positions = []
                sensor_rotations = []
                projection_indices = []
                masks_outliers = []

                features_encoder = []
                features_lastlayer =  np.zeros((20, 64, 240, 320), dtype=np.float32)
                features_scores = []

                detection_data = []
                segmentation_data = []

                with torch.no_grad():
                    for n in tqdm(range(nb_frames_per_sample)):
                        pos = sub_path['positions'][n]
                        ori = sub_path['orientations'][n]

                        habitat.position = list(pos)
                        habitat.rotation = list(ori)
                        habitat.set_agent_state()

                        sensor_pos = habitat.get_sensor_pos()
                        sensor_ori = habitat.get_sensor_ori()

                        sensor_positions.append(sensor_pos)
                        sensor_rotations.append([sensor_ori.x, sensor_ori.y, sensor_ori.z, sensor_ori.w])

                        # -- get T transorm
                        sensor_ori = np.array([sensor_ori.x, sensor_ori.y, sensor_ori.z, sensor_ori.w])
                        r = R.from_quat(sensor_ori)
                        elevation, heading, bank = r.as_rotvec()

                        xyzhe = np.array([[sensor_pos[0],
                                        sensor_pos[1],
                                        sensor_pos[2],
                                        heading,
                                        elevation + np.pi]])
                        xyzhe = torch.FloatTensor(xyzhe).to(device)
                        T = _transform3D(xyzhe, device=device)

                        # -- depth for projection
                        depth = habitat.render(mode='depth')
                        depth = depth[:,:,0]
                        depth = depth[np.newaxis,...]
                        frames_depth.append(depth)
                        depth = habitat.render(mode='depth')
                        depth = depth[:,:,0]
                        depth = depth.astype(np.float32)
                        depth *= 10.0
                        depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(device)

                        pc, mask = projector.forward(depth_var, T)

                        pc = pc.cpu().numpy()
                        mask_outliers = mask.cpu().numpy()
                        projection_indices.append(pc)
                        masks_outliers.append(mask_outliers)

                        # -- get semantic labels
                        rgb = habitat.render()
                        rgb = rgb[np.newaxis,...]
                        frames_RGB.append(rgb)
                        rgb = habitat.render()
                        rgb = rgb.astype(np.float32)
                        rgb = rgb / 255.0
                        rgb = torch.FloatTensor(rgb).permute(2,0,1)
                        rgb = normalize(rgb)
                        rgb = rgb.unsqueeze(0).to(device)

                        depth_enc = habitat.render(mode='depth')
                        depth_enc = depth_enc[:,:,0]
                        depth_enc = depth_enc.astype(np.float32)
                        depth_enc = torch.FloatTensor(depth_enc).unsqueeze(0)
                        depth_enc = depth_normalize(depth_enc)
                        depth_enc = depth_enc.unsqueeze(0).to(device)

                        # -- get egocentric features
                        rgb = habitat.render()
                        # switch rbg to bgr
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                        
                        # get detections and segmentation gt
                        detections_n = habitat.render_bbox_lvis_20()

                        # save detection data required to train the detector
                        image_data = {'file_name': house + '_' + level + '_' + str(start+n) + '.jpg', 'image': rgb, 'height': 480, 'width': 640, 'gt_boxes': [det['bbox'] for det in detections_n], 'gt_classes': [det['category_id'] for det in detections_n]}
                        detection_data.append(str(image_data))

                        # save segmentation data required to train the segmentation network
                        segmentation_n = habitat.render_semantic_lvis_20() # segmentations are shifted by 1 relative to detections to account for void class

                        # iterate through each pixel
                        # sem_vis = rgb.copy()
                        # for i in range(segmentation_n.shape[0]):
                        #     for j in range(segmentation_n.shape[1]):
                        #         # get the semantic label
                        #         label = segmentation_n[i,j]
                        #         # if the label is not 0 (background)
                        #         if label != 0:
                        #             # set the rgb value to the corresponding color
                        #             sem_vis[i,j] = palette[label]
                        # # display image with opencv
                        # cv2.imshow('semantic', sem_vis)
                        # cv2.waitKey(0)

                        segmentation_data.append(segmentation_n)

                frames_RGB = np.concatenate(frames_RGB, axis=0)
                frames_depth = np.concatenate(frames_depth, axis=0)
                sensor_positions = np.array(sensor_positions)
                sensor_rotations = np.array(sensor_rotations)
                masks_outliers = np.concatenate(masks_outliers, axis=0)
                projection_indices = np.concatenate(projection_indices, axis=0)

                segmentation_data = np.array(segmentation_data)

                with h5py.File(filename, 'w') as f:
                    f.create_dataset('rgb', data=frames_RGB, dtype=np.uint8)
                    f.create_dataset('depth', data=frames_depth, dtype=np.float32)
                    f.create_dataset('sensor_positions', data=sensor_positions, dtype=np.float32)
                    f.create_dataset('sensor_rotations', data=sensor_rotations, dtype=np.float32)
                    f.create_dataset('projection_indices', data=projection_indices, dtype=np.float32)
                    f.create_dataset('masks_outliers', data=masks_outliers, dtype=np.bool)
                    #f.create_dataset('features_encoder', data=features_encoder, dtype=np.float32)
                    f.create_dataset('features_lastlayer', data=features_lastlayer, dtype=np.float32)
                    #f.create_dataset('features_scores', data=features_scores, dtype=np.float32)
                    f.create_dataset('detection_data', data=detection_data, dtype=h5py.special_dtype(vlen=str))
                    f.create_dataset('segmentation_data', data=segmentation_data, dtype=np.uint8)

            del habitat




