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
from semseg.rednet import RedNet

from utils.habitat_utils import HabitatUtils
from utils import convert_weights_cuda_cpu

from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms

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


# -- settings
output_dir = 'data/training/replica_data/'
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

paths = json.load(open('data_replica/paths.json', 'r'))


device = torch.device('cuda')

# -- Create model
# -- instantiate RedNet
cfg_rednet = {
    'arch': 'rednet',
    'resnet_pretrained': False,
    'finetune': True,
    'SUNRGBD_pretrained_weights': '',
    'n_classes': 13,
    'upsample_prediction': True,
    'model_path': 'pretrained_models/rednet_mp3d_best_model.pkl',
}
model = RedNet(cfg_rednet)
model = model.to(device)

print('Loading pre-trained weights: ', cfg_rednet['model_path'])
state = torch.load(cfg_rednet['model_path'])
model_state = state['model_state']
model_state = convert_weights_cuda_cpu(model_state, 'cpu')
model.load_state_dict(model_state)
model = model.eval()

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
# with open('data/envs_splits.json', 'r') as f:
#     env_splits = json.load(f)

"""
 -->> START
"""
info = {}
map_info = {}
count = 0
for env, path in tqdm(paths.items()):

    count+=1

    # if env in ['YFuZgdQ5vWj_0', 'X7HyMhZNoso_0', 'Vvot9Ly1tCj_0']:
    #     pass
    # else:
    #     continue

    
    if env.split('_')[0] == 'frl':
        house = env.split('_')[0] + '_' + env.split('_')[1] + '_' + env.split('_')[2]
        if len(env.split('_')) == 4:
            level = env.split('_')[4]
        else:
            level = '0'
    else: 
        house = env.split('_')[0] + '_' + env.split('_')[1]
        if len(env.split('_')) == 3:
            level = env.split('_')[2]
        else:
            level = '0'

    print(house, level)
    # scene = 'data/mp3d/{}/{}.glb'.format(house, house)
    # scene = '../Matterport/habitat_data/v1/tasks/mp3d/{}/{}.glb'.format(house, house)
    scene = '../Replica-Dataset/data/{}/mesh.ply'.format(house)


    habitat = HabitatUtils(scene, level, housetype='replica')

    N = len(path['positions'])

    info[env] = {}

    # max and min points on the map
    x_range = [100, -100]
    y_range = [100, -100]

    for m in range(nb_samples_per_env):

        start = np.random.randint(0, high=N-nb_frames_per_sample)

        info[env][m] = {'start':start}

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
        features_lastlayer = []
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

                # update the bounds of the map
                max_x = torch.max(pc[0,:,:,0])
                min_x = torch.min(pc[0,:,:,0])
                max_y = torch.max(pc[0,:,:,2])
                min_y = torch.min(pc[0,:,:,2])
                if max_x > x_range[1]:
                    x_range[1] = max_x
                if min_x < x_range[0]:
                    x_range[0] = min_x
                if max_y > y_range[1]:
                    y_range[1] = max_y
                if min_y < y_range[0]:
                    y_range[0] = min_y

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

                semfeat_lastlayer= model(rgb, depth_enc)
                semfeat_lastlayer = semfeat_lastlayer.cpu().numpy()
                features_lastlayer.append(semfeat_lastlayer)
                
                # -- get egocentric features
                rgb = habitat.render()
                # switch rbg to bgr
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                # display image with opencv
                # cv2.imshow('rgb', rgb)
                # cv2.waitKey(0)
                
                # # get detections and segmentation gt
                detections_n = habitat.render_bbox_lvis_replica()

                # save detection data required to train the detector
                image_data = {'file_name': house + '_' + level + '_' + str(start+n) + '.jpg', 'image': rgb, 'height': 480, 'width': 640, 'gt_boxes': [det['bbox'] for det in detections_n], 'gt_classes': [det['category_id'] for det in detections_n]}
                detection_data.append(str(image_data))

                # save segmentation data required to train the segmentation network
                segmentation_n = habitat.render_semantic_lvis_replica() # segmentations are shifted by 1 relative to detections to account for void class

                # # iterate through each pixel
                # sem_vis = rgb.copy()
                # for i in range(segmentation_n.shape[0]):
                #     for j in range(segmentation_n.shape[1]):
                #         # get the semantic label
                #         label = segmentation_n[i,j]
                #         # if the label is not 0 (background)
                #         if label != 0:
                #             # set the rgb value to the corresponding color
                #             sem_vis[i,j] = palette[label]
                # for box in image_data['gt_boxes']:
                #     # draw red box around object
                #     cv2.rectangle(sem_vis, (int(box[0]), int(box[1])), (int(box[2])+int(box[0]), int(box[3])+int(box[1])), (255,0,0), 2)

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

        features_lastlayer = np.concatenate(features_lastlayer, axis=0)

        segmentation_data = np.array(segmentation_data)

        filename = os.path.join(output_dir, env+'_{}.h5'.format(m))
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

    map_info[env] = {'x_min': x_range[0].item(), 'x_max': x_range[1].item(), 'y_min': y_range[0].item(), 'y_max': y_range[1].item()}
    print(map_info[env])
    del habitat

json.dump(info, open('data/training/info_replica_data.json', 'w'))
json.dump(map_info, open('data/training/replica_map_info.json', 'w'))



