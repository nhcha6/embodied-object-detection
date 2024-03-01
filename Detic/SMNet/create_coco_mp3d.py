import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2

from projector import _transform3D
from projector.projector import Projector
from scipy.spatial.transform import Rotation as R

from utils.habitat_utils import HabitatUtils

from utils.semantic_utils import use_fine, object_whitelist, object_lvis
import argparse

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Create COCO dataset')
    parser.add_argument('--data_path', type=str, default='./', help='path to Matterport data')
    args = parser.parse_args()

    # train, val or test
    splits = ['train', 'val', 'test']
    for split in splits:

        # if split != 'val':
            # continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Settings
        resolution = 0.02 # topdown resolution
        default_ego_dim = (480, 640) #egocentric resolution
        z_clip = 0.50 # detections over z_clip will be ignored
        vfov = 67.5
        vfov = vfov * np.pi / 180.0

        # -- load JSONS 
        info = json.load(open('./semmap_GT_info.json','r'))
        paths = json.load(open('./paths.json', 'r'))

        # image counters
        image_id = 0
        det_id = 0

        # coco annotations in json format
        coco_anno = {}
        coco_anno["images"] = []
        coco_anno["annotations"] = []
        coco_anno["categories"] = []

        # add categories
        for i, obj in enumerate(object_lvis):
            coco_anno["categories"].append({"id": i, "name": obj})

        # make relevant directories
        SAVE_DIR = f'../embodied_data/mp3d_{split}/'

        # images
        if not os.path.exists(os.path.join(SAVE_DIR, 'JPEGImages')):
            os.makedirs(os.path.join(SAVE_DIR, 'JPEGImages'))

        # load data/env_splits.json
        with open('./envs_splits.json', 'r') as f:
            env_splits = json.load(f)
        envs = env_splits[split+ '_envs']

        # iterate through each environment
        for env in envs:
            # env = '17DRP5sb8fy_0'
            
            # -- instantiate Habitat

            house, level = env.split('_')

            # -- get house info
            try:
                world_dim_discret = info[env]['dim']
                map_world_shift = info[env]['map_world_shift']
                map_world_shift = np.array(map_world_shift)
                world_shift_origin=torch.from_numpy(map_world_shift).float().to(device=device)
            except:
                continue

            # args.data_path should contain the path to the downloaded Matterport data
            scene = os.path.join(args.data_path, 'habitat_data/v1/tasks/mp3d/{}/{}.glb'.format(house, house))

            # habitat = HabitatUtils(scene, int(level), housetype='replica')
            habitat = HabitatUtils(scene, int(level))

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

            normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

            depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

            # compute projections indices and egocentric features
            path = paths[env]

            N = len(path['positions'])

            projections_wtm = np.zeros((N,480,640,2), dtype=np.uint16)
            projections_masks = np.zeros((N,480,640), dtype=np.bool)
            projections_heights = np.zeros((N,480,640), dtype=np.float32)
            depths = np.zeros((N,480,640), dtype=np.float32)
            # features_lastlayer = np.zeros((N,64,240,320), dtype=np.float32)

            print('Compute egocentric features and projection indices')

            with torch.no_grad():
                for n in tqdm(range(N)):

                    pos = path['positions'][n]
                    ori = path['orientations'][n]

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

                    # -- get egocentric features
                    rgb = habitat.render()
                    # switch rbg to bgr
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    detections_n = habitat.render_bbox_lvis_20()

                    # iterate through each detection
                    # image = np.ascontiguousarray(rgb, dtype=np.uint8)
                    # for det in detections:
                    #     # get the semantic label
                    #     label = det['category_id']
                    #     bbox = det['bbox']
                    #     # draw rectangle
                    #     cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3]+bbox[1], bbox[2]+bbox[0]), tuple(int(x) for x in palette[label]), 2)
                    #     cv2.putText(image, object_lvis[label], (bbox[1], bbox[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(int(x) for x in palette[label]), 2)

                    # # display image with opencv
                    # cv2.imshow('rgb', image)
                    # cv2.waitKey(0)

                    if n%5 == 0:
                        # add detections
                        for d in detections_n:
                            d['id'] = det_id
                            det_id += 1
                            d['image_id'] = image_id
                            coco_anno["annotations"].append(d)

                        # add image
                        coco_anno["images"].append({"id": image_id, "file_name": os.path.join('JPEGImages', env + '_' + str(n) + '.jpg'), "height": 480, "width": 640})
                        image_id += 1

                    # save rgb image
                    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                    cv2.imwrite(os.path.join(SAVE_DIR, 'JPEGImages', env + '_' + str(n) + '.jpg'), rgb)

            # save coco annotations
            with open(os.path.join(SAVE_DIR, 'annotations.json'), 'w') as outfile:
                json.dump(coco_anno, outfile)

            # save projections, heights, masks, depths
            # np.save(os.path.join(SAVE_DIR, f'{env}_projections_wtm.npy'), projections_wtm)
            # np.save(os.path.join(SAVE_DIR, f'{env}_projections_masks.npy'), projections_masks)
            # np.save(os.path.join(SAVE_DIR, f'{env}_projection_heights.npy'), projections_heights)
            # np.save(os.path.join(SAVE_DIR, f'{env}_depths.npy'), depths)

            del habitat, projector, projections_wtm, projections_masks, projections_heights, depths

        # save coco annotations
        with open(os.path.join(SAVE_DIR, 'annotations.json'), 'w') as outfile:
            json.dump(coco_anno, outfile)

   