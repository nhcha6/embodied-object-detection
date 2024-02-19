import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2

from SMNet.model_test import SMNet

from projector import _transform3D
from projector.projector import Projector
from scipy.spatial.transform import Rotation as R

from utils.habitat_utils import HabitatUtils
from utils import convert_weights_cuda_cpu

from semseg.rednet import RedNet

from utils.semantic_utils import use_fine, object_whitelist, object_lvis

# env = '17DRP5sb8fy_0'
env = '1LXtFkjw3qL_0'

print(env)

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
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255]
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Settings
resolution = 0.02 # topdown resolution
default_ego_dim = (480, 640) #egocentric resolution
z_clip = 0.50 # detections over z_clip will be ignored
vfov = 67.5
vfov = vfov * np.pi / 180.0

# -- load JSONS 
info = json.load(open('data/semmap_GT_info.json','r'))
paths = json.load(open('data/paths.json', 'r'))
 
# -- instantiate Habitat

house, level = env.split('_')
# scene = 'data/mp3d/{}/{}.glb'.format(house, house)
scene = '../Matterport/habitat_data/v1/tasks/mp3d/{}/{}.glb'.format(house, house)
# scene = '/home/nicolas/hpc-home/allocentric_memory/Replica-Dataset/data/apartment_0/habitat/mesh_preseg_semantic.ply'

print(scene)
# habitat = HabitatUtils(scene, int(level), housetype='replica')
habitat = HabitatUtils(scene, int(level))
print('done')

# -- get house info
world_dim_discret = info[env]['dim']
map_world_shift = info[env]['map_world_shift']
map_world_shift = np.array(map_world_shift)
world_shift_origin=torch.from_numpy(map_world_shift).float().to(device=device)

# -- instantiate projector
projector = Projector(vfov, 1,
                      default_ego_dim[0],
                      default_ego_dim[1],
                      world_dim_discret[2], # height
                      world_dim_discret[0], # width
                      resolution,
                      world_shift_origin,
                      z_clip,
                      device=device)

# # -- Create RedNet model
# cfg_rednet = {
#     'arch': 'rednet',
#     'resnet_pretrained': False,
#     'finetune': True,
#     'SUNRGBD_pretrained_weights': '',
#     'n_classes': 13,
#     'upsample_prediction': True,
#     'load_model': 'rednet_mp3d_best_model.pkl',
# }

# model_rednet = RedNet(cfg_rednet)
# model_rednet = model_rednet.to(device)

# print('Loading pre-trained weights: ', cfg_rednet['load_model'])
# state = torch.load(cfg_rednet['load_model'])
# model_state = state['model_state']
# model_state = convert_weights_cuda_cpu(model_state, 'cpu')
# model_rednet.load_state_dict(model_state)
# model_rednet.eval()


normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])


# compute projections indices and egocentric features
path = paths[env]

N = len(path['positions'])

projections_wtm = np.zeros((N,480,640,2), dtype=np.uint16)
projections_masks = np.zeros((N,480,640), dtype=np.bool)
projections_heights = np.zeros((N,480,640), dtype=np.float32)

features_lastlayer = np.zeros((N,64,240,320), dtype=np.float32)

print('Compute egocentric features and projection indices')

with torch.no_grad():
    for n in tqdm(range(N)):
        pos = path['positions'][n]
        ori = path['orientations'][n]
        print(pos)
        print(ori)

        habitat.position = list(pos)
        habitat.rotation = list(ori)
        habitat.set_agent_state()

        sensor_pos = habitat.get_sensor_pos()
        sensor_ori = habitat.get_sensor_ori()

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
        depth = depth.astype(np.float32)
        depth *= 10.0
        depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(device)

        # -- projection
        world_to_map, mask_outliers, heights = projector.forward(depth_var, T, return_heights=True)

        world_to_map = world_to_map[0].cpu().numpy()
        mask_outliers = mask_outliers[0].cpu().numpy()
        heights = heights[0].cpu().numpy()

        world_to_map = world_to_map.astype(np.uint16)
        mask_outliers = mask_outliers.astype(np.bool)
        heights = heights.astype(np.float32)

        projections_wtm[n,...] = world_to_map
        projections_masks[n,...] = mask_outliers
        projections_heights[n,...] = heights
 
        # -- get egocentric features
        rgb = habitat.render()
        # switch rbg to bgr
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(rgb, dtype=np.uint8)
        detections = habitat.render_bbox_lvis_20()

        # iterate through each detection
        for det in detections:
            # get the semantic label
            label = det['category_id']
            bbox = det['bbox']
            # draw rectangle
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2]+bbox[0], bbox[3]+bbox[1]), tuple(int(x) for x in palette[label]), 2)
            cv2.putText(image, object_lvis[label], (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(int(x) for x in palette[label]), 2)

        # display image with opencv
        cv2.imshow('rgb', image)


        semantic = habitat.render_semantic_lvis_20()
        # iterate through each pixel
        for i in range(semantic.shape[0]):
            for j in range(semantic.shape[1]):
                # get the semantic label
                label = semantic[i,j]
                # if the label is not 0 (background)
                if label != 0:
                    # set the rgb value to the corresponding color
                    rgb[i,j] = palette[label]
        

        # display image with opencv
        cv2.imshow('semantic', rgb)

        # -- depth for projection
        depth_enc = habitat.render(mode='depth')
        cv2.imshow('depth', depth_enc)

        # print(world_to_map)
        # print(mask_outliers)
        # print(heights)

        cv2.waitKey(0)

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


        # semfeat_lastlayer = model_rednet(rgb, depth_enc)
        # semfeat_lastlayer = semfeat_lastlayer[0].cpu().numpy()
        # semfeat_lastlayer = semfeat_lastlayer.astype(np.float32)
        # features_lastlayer[n,...] = semfeat_lastlayer

del habitat, projector

