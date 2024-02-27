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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str, default='./', help='path to Replica data')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Settings
    resolution = 0.02 # topdown resolution
    default_ego_dim = (480, 640) #egocentric resolution
    z_clip = 0.50 # detections over z_clip will be ignored
    vfov = 67.5
    vfov = vfov * np.pi / 180.0

    # -- load JSONS 
    # info = json.load(open('data/semmap_GT_info.json','r'))
    paths = json.load(open('./replica_paths.json', 'r'))

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
    SAVE_DIR = f'../embodied_data/replica/'

    # images
    if not os.path.exists(os.path.join(SAVE_DIR, 'JPEGImages')):
        os.makedirs(os.path.join(SAVE_DIR, 'JPEGImages'))

    # iterate through each environment
    for env, path in tqdm(paths.items()):
        
        # -- instantiate Habitat

        print(env)

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

        scene = os.path.join(args.data_path, 'data/{}/mesh.ply'.format(house))

        habitat = HabitatUtils(scene, level, housetype='replica')

        N = len(path['positions'])

        with torch.no_grad():
            for n in tqdm(range(N)):

                pos = path['positions'][n]
                ori = path['orientations'][n]

                habitat.position = list(pos)
                habitat.rotation = list(ori)
                habitat.set_agent_state()

                sensor_pos = habitat.get_sensor_pos()
                sensor_ori = habitat.get_sensor_ori()

                # -- get egocentric features
                rgb = habitat.render()
                # switch rbg to bgr
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                detections_n = habitat.render_bbox_lvis_replica()

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
                cv2.imwrite(os.path.join(SAVE_DIR, 'JPEGImages', house + '_' + level + '_' + str(n) + '.jpg'), rgb)

        # save coco annotations
        with open(os.path.join(SAVE_DIR, 'annotations.json'), 'w') as outfile:
            json.dump(coco_anno, outfile)

        del habitat

    # save coco annotations
    with open(os.path.join(SAVE_DIR, 'annotations.json'), 'w') as outfile:
        json.dump(coco_anno, outfile)