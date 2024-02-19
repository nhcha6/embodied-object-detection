# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss

import socket
import pickle
import struct ## new

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo, EmbodiedVisualizationDemo
import torch
import os
import numpy as np
import cv2
import torch
import math
from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from PIL import Image

def _transform3D(xyzhe, device=torch.device("cpu")):
    """ Return (N, 4, 4) transformation matrices from (N,5) x,y,z,heading,elevation """

    theta_x = xyzhe[:,4] # elevation
    cx = torch.cos(theta_x)
    sx = torch.sin(theta_x)

    theta_y = xyzhe[:,3] # heading
    cy = torch.cos(theta_y)
    sy = torch.sin(theta_y)

    T = torch.zeros(xyzhe.shape[0], 4, 4, device=device)
    T[:,0,0] =  cy
    T[:,0,1] =  sx*sy
    T[:,0,2] =  cx*sy 
    T[:,0,3] =  xyzhe[:,0] # x

    T[:,1,0] =  0
    T[:,1,1] =  cx
    T[:,1,2] =  -sx
    T[:,1,3] =  xyzhe[:,1] # y

    T[:,2,0] =  -sy
    T[:,2,1] =  cy*sx
    T[:,2,2] =  cy*cx
    T[:,2,3] =  xyzhe[:,2] # z

    T[:,3,3] =  1

    R = torch.zeros(xyzhe.shape[0], 4, 4, device=device)
    R[:,0,0] =  0
    R[:,0,1] =  0
    R[:,0,2] =  1
    R[:,0,3] =  0

    R[:,1,0] =  0
    R[:,1,1] =  1
    R[:,1,2] =  0
    R[:,1,3] =  0

    R[:,2,0] =  1
    R[:,2,1] =  0
    R[:,2,2] =  0
    R[:,2,3] =  0

    R[:,3,3] =  1

    # multiply matrices together
    T = torch.matmul(T,R)

    return T

class ProjectorUtils():
    def __init__(self,
                vfov,
                hfov,
                batch_size,
                feature_map_height,
                feature_map_width,
                output_height,
                output_width,
                gridcellsize,
                world_shift_origin,
                z_clip_threshold,
                device,
                ):
            
        self.vfov = vfov
        self.hfov = hfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.output_height = output_height # dimensions of the topdown map
        self.output_width = output_width
        self.gridcellsize = gridcellsize
        self.z_clip_threshold = z_clip_threshold
        self.world_shift_origin = world_shift_origin
        self.device = device

        self.x_scale, self.y_scale, self.ones = self.compute_scaling_params(
            batch_size, feature_map_height, feature_map_width
        )

    
    def compute_intrinsic_matrix(self, width, height, vfov):
        K = torch.tensor([[380.3127746582031, 0.0, 315.81829833984375], [0.0, 379.828857421875, 250.9555206298828], [0, 0, 1]])
        return K


    def compute_scaling_params(self, batch_size, image_height, image_width):
        """ Precomputes tensors for calculating depth to point cloud """
        # (float tensor N,3,3) : Camera intrinsics matrix
        K = self.compute_intrinsic_matrix(image_width, image_height, self.vfov)
        K = K.to(device=self.device).unsqueeze(0)
        K = K.expand(batch_size, 3, 3)

        fx = K[:, 0, 0].unsqueeze(1).unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1).unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1).unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1).unsqueeze(1)

        x_rows = torch.arange(start=0, end=image_width, device=self.device)
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((image_height, 1))
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((batch_size, 1, 1))
        x_rows = x_rows.float()

        y_cols = torch.arange(start=0, end=image_height, device=self.device)
        y_cols = y_cols.unsqueeze(1)
        y_cols = y_cols.repeat((1, image_width))
        y_cols = y_cols.unsqueeze(0)
        y_cols = y_cols.repeat((batch_size, 1, 1))
        y_cols = y_cols.float()

        # 0.5 is so points are projected through the center of pixels
        x_scale = (x_rows + 0.5 - cx) / fx#; print(x_scale[0,0,:])
        y_scale = (y_cols + 0.5 - cy) / fy#; print(y_scale[0,:,0]); stop
        ones = (
            torch.ones((batch_size, image_height, image_width), device=self.device)
            .unsqueeze(3)
            .float()
        )
        return x_scale, y_scale, ones

    def point_cloud(self, depth, depth_scaling=1.0):
        """
        Converts image pixels to 3D pointcloud in camera reference using depth values.

        Args:
            depth (torch.FloatTensor): (batch_size, height, width)

        Returns:
            xyz1 (torch.FloatTensor): (batch_size, height * width, 4)

        Operation:
            z = d / scaling
            x = z * (u-cx) / fx
            y = z * (v-cv) / fy
        """
        shape = depth.shape
        if (
            shape[0] == self.batch_size
            and shape[1] == self.fmh
            and shape[2] == self.fmw
        ):
            x_scale = self.x_scale
            y_scale = self.y_scale
            ones = self.ones
        else:
            x_scale, y_scale, ones = self.compute_scaling_params(
                shape[0], shape[1], shape[2]
            )
        z = depth / float(depth_scaling)
        x = z * x_scale
        y = z * y_scale

        xyz1 = torch.cat((x.unsqueeze(3), y.unsqueeze(3), z.unsqueeze(3), ones), dim=3)
        return xyz1

    def transform_camera_to_world(self, xyz1, T):
        """
        Converts pointcloud from camera to world reference.

        Args:
            xyz1 (torch.FloatTensor): [(x,y,z,1)] array of N points in homogeneous coordinates
            T (torch.FloatTensor): camera-to-world transformation matrix
                                        (inverse of extrinsic matrix)

        Returns:
            (float tensor BxNx4): array of pointcloud in homogeneous coordinates

        Shape:
            Input:
                xyz1: (batch_size, 4, no_of_points)
                T: (batch_size, 4, 4)
            Output:
                (batch_size, 4, no_of_points)

        Operation: T' * R' * xyz
                   Here, T' and R' are the translation and rotation matrices.
                   And T = [R' T'] is the combined matrix provided in the function as input
                           [0  1 ]
        """
        return torch.bmm(T, xyz1)

    def pixel_to_world_mapping(self, depth_img_array, T):
        """
        Computes mapping from image pixels to 3D world (x,y,z)

        Args:
            depth_img_array (torch.FloatTensor): Depth values tensor
            T (torch.FloatTensor): camera-to-world transformation matrix (inverse of
                                        extrinsic matrix)

        Returns:
            pixel_to_world (torch.FloatTensor) : Mapping of one image pixel (i,j) in 3D world
                                                      (x,y,z)
                    array cell (i,j) : (x,y,z)
                        i,j - image pixel indices
                        x,y,z - world coordinates

        Shape:
            Input:
                depth_img_array: (N, height, width)
                T: (N, 4, 4)
            Output:
                pixel_to_world: (N, height, width, 3)
        """

        # Transformed from image coordinate system to camera coordinate system, i.e origin is
        # Camera location  # GEO:
        # shape: xyz1 (batch_size, height, width, 4)
        xyz1 = self.point_cloud(depth_img_array)

        # shape: (batch_size, height * width, 4)
        xyz1 = torch.reshape(xyz1, (xyz1.shape[0], xyz1.shape[1] * xyz1.shape[2], 4))

        # shape: (batch_size, 4, height * width)
        xyz1_t = torch.transpose(xyz1, 1, 2)  # [B,4,HxW]
        # world_xyz = xyz1_t.transpose(1, 2)[:, :, :3]

        # Transformed points from camera coordinate system to world coordinate system  # GEO:
        # shape: xyz1_w(batch_size, 4, height * width)
        xyz1_w = self.transform_camera_to_world(xyz1_t, T)

        # shape: (batch_size, height * width, 3)
        world_xyz = xyz1_w.transpose(1, 2)[:, :, :3]

        # -- shift world origin
        world_xyz -= self.world_shift_origin

        # shape: (batch_size, height, width, 3)
        pixel_to_world = torch.reshape(world_xyz,((depth_img_array.shape[0],depth_img_array.shape[1],depth_img_array.shape[2],3,)),)

        return pixel_to_world

    def discretize_point_cloud(self, point_cloud, camera_height):
        """ #GEO:
        Maps pixel in world coordinates to an (output_height, output_width) map.
        - Discretizes the (x,y) coordinates of the features to gridcellsize.
        - Remove features that lie outside the (output_height, output_width) size.
        - Computes the min_xy and size_xy, and change (x,y) coordinates setting min_xy as origin.

        Args:
            point_cloud (torch.FloatTensor): (x,y,z) coordinates of features in 3D world
            camera_height (torch.FloatTensor): y coordinate of the camera used for deciding
                                                      after how much height to crop

        Returns:
            pixels_in_map (torch.LongTensor): World (x,y) coordinates of features discretized
                                    in gridcellsize and cropped to (output_height, output_width).

        Shape:
            Input:
                point_cloud: (batch_size, features_height, features_width, 3)
                camera_height: (batch_size)
            Output:
                pixels_in_map: (batch_size, features_height, features_width, 2)
        """
        
        # -- /!\/!\
        # -- /!\ in Habitat-MP3D y-axis is up. /!\/!\
        # -- /!\/!\
        pixels_in_map = ((point_cloud[:, :, :, [0,2]])/ self.gridcellsize).round()
        
        # Anything outside map boundary gets mapped to (0,0) with an empty feature
        # mask for outside map indices
        outside_map_indices = (pixels_in_map[:, :, :, 0] >= self.output_width) +\
                              (pixels_in_map[:, :, :, 1] >= self.output_height) +\
                              (pixels_in_map[:, :, :, 0] < 0) +\
                              (pixels_in_map[:, :, :, 1] < 0)
        
        # shape: camera_y (batch_size, features_height, features_width)
        camera_y = (camera_height.unsqueeze(1).unsqueeze(1).repeat(1, pixels_in_map.shape[1], pixels_in_map.shape[2]))
        
        # Anything above camera_y + z_clip_threshold will be ignored
        above_threshold_z_indices = point_cloud[:, :, :, 1] > (camera_y + self.z_clip_threshold)
        
        mask_outliers = outside_map_indices + above_threshold_z_indices

        return pixels_in_map.long(), mask_outliers

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    # cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.INPUT.MIN_SIZE_TEST = 300
    
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom', 'mp3d'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    # set-up the model
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = EmbodiedVisualizationDemo(cfg, args)

    # embodied imports
    data_path = "/home/nicolas/Documents/jackle_data/"
    device = torch.device("cuda")

    map_world_shift = np.zeros(3)
    world_shift_origin=torch.from_numpy(map_world_shift).float().to(device=device)
    vfov = 58*np.pi / 180.0
    hfov = 87*np.pi / 180.0

    # projector
    projector = ProjectorUtils(vfov=vfov,
                    hfov=hfov,
                    batch_size=1,
                    feature_map_height=480,
                    feature_map_width=640,
                    output_height=1,
                    output_width=1,
                    gridcellsize=1,
                    world_shift_origin = world_shift_origin,
                    z_clip_threshold=3,
                    device=device)

    # memory grid
    res = 0.2
    map_w, map_h = 40, 40
    map_h = math.ceil(map_h / res)
    map_w = math.ceil(map_w / res)
    empty_map = np.zeros((map_h*map_w, 3))
    map_world_shift = torch.FloatTensor([-13, 0, -13]).to(device)

    # depth to range
    # depth_to_range = np.zeros((480,640))
    # for i in range(480):
    #     for j in range(640):
    #         depth_to_range[i,j] = math.cos(abs(i-240)/480 * vfov)*math.cos(abs(j-320)/640 * hfov)

    colours = [(0,0,255), (0,255,0), (255,0,0)]
    count = -1
    img_count = 0
    for folder in os.listdir(data_path):
        if os.path.isdir(data_path+folder):
            print(data_path+folder)
            images = sorted(os.listdir(data_path+folder + "/images"))
            depth = sorted(os.listdir(data_path+folder + "/depth"))
            pose = sorted(os.listdir(data_path + folder + "/pose"))
            # iterate through each rgb image, sampling at approx 10Hz. For each rgb image, find the closest depth image and odometry data.
            for image in images[::2]:

                ############# FIND THE CLOSEST DEPTH IMAGE AND POSE #############

                time = image.split(".")[0]
                # find the depth image with the closest time
                closest_depth = min(depth, key=lambda x:abs(int(x.split(".")[0]) - int(time)))
                # find the pose with the closest time
                closest_pose = min(pose, key=lambda x:abs(int(x.split(".")[0]) - int(time)))

                depth_image = cv2.imread(data_path + folder + "/depth/" + closest_depth, cv2.IMREAD_ANYDEPTH)
                pose_val = np.load(data_path + folder + "/pose/" + closest_pose)

                # rgb_image = cv2.imread(data_path + folder + "/images/" + image)
                # manually import the image, same as done in original DETIC dataloader
                with PathManager.open(data_path + folder + "/images/" + image, "rb") as f:
                    rgb_image = Image.open(f)
                    # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
                    rgb_image = _apply_exif_orientation(rgb_image)
                    rgb_image = convert_PIL_to_numpy(rgb_image, 'RGB')

                ############ PROJECT IMAGE ONTO MAP #############

                # convert depth to format for projection
                depth_var = depth_image/1000
                # print(depth_var[240, 480])
                depth_var = torch.FloatTensor(depth_var).unsqueeze(0).to(device=device)

                xyzhe = np.array([[pose_val[0], 0.65, pose_val[1], -1*pose_val[2], np.pi+0.06]])
                xyzhe = torch.FloatTensor(xyzhe).to(device)
                T = _transform3D(xyzhe, device=device)

                point_cloud = projector.pixel_to_world_mapping(depth_var, T)

                # calculate projection of the point
                point_cloud -= map_world_shift
                pixels_in_map = (point_cloud[:,:,:, [0,2]]/res).round().long()
                pixels_in_map = pixels_in_map.cpu().numpy()
                pixels_in_map[:,:,:,1] = np.clip(pixels_in_map[:,:,:,1], 0, map_h-1)
                pixels_in_map[:,:,:,0] = np.clip(pixels_in_map[:,:,:,0], 0, map_w-1)  

                # clip projection indices to map size and flatten  
                proj_indices = pixels_in_map[:,:,:,0] * map_h + pixels_in_map[:,:,:,1]
                proj_indices = proj_indices[:, :, :,np.newaxis].squeeze(0)
                
                # also put the robot pos on the map
                shifted_pose = torch.FloatTensor(pose_val[[0,1]]).to(device) - map_world_shift[[0,2]]
                robot_pos = (shifted_pose/res).round().long()

                ############# VISUALISE ####################

                rgb = rgb_image.copy()
                # draw dot on rgb_image at 320,320
                cv2.circle(rgb, (480,240), 5, (0,0,255), -1)

                # iterate through each pixel and update the map
                # if pixels_in_map[0,240,480,0] < 40/res and pixels_in_map[0,240,480,1] < 40/res:
                    # empty_map[pixels_in_map[0,240,480,0], pixels_in_map[0,240,480,1]] = colours[count]
                    # empty_map[robot_pos[0], robot_pos[1]] = (0,165,255)
                
                # update map
                empty_map[proj_indices[240,480,0]] = colours[count]
                empty_map[robot_pos[0]*map_h + robot_pos[1]] = (0,165,255)

                # # show map and set to dynamic window size
                # cv2.namedWindow('map', cv2.WINDOW_NORMAL)
                # cv2.imshow("map", empty_map.reshape(map_h, map_w, 3))
                
                # # show image
                # cv2.namedWindow('Image')
                # cv2.imshow("Image", rgb)
                # cv2.imshow("depth", depth_image)
                # cv2.waitKey(0)

                ############### GENERATE OBJECT DETECTION DATA ##################

                # create empty memory
                memory = np.zeros((map_w*map_h, 256))

                inputs = {"image": rgb_image, "proj_indices": proj_indices, "memory_reset": img_count==0, "sequence_name": folder, "memory": memory}
                img_count += 1

                predictions, visualized_output = demo.run_on_data(inputs)
                print(predictions)

                cv2.namedWindow('ImageWindow',cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('image', 600,600)
                cv2.imshow('ImageWindow',visualized_output.get_image()[:, :, ::-1])
                cv2.waitKey(0)