# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import argparse

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.config import add_detic_config
from detic.data.custom_build_augmentation import build_custom_augmentation
from detic.data.custom_dataset_dataloader import  build_custom_train_loader
from detic.data.custom_dataset_mapper import CustomDatasetMapper, DetrDatasetMapper
from detic.custom_solver import build_custom_optimizer
from detic.evaluation.oideval import OIDEvaluator
from detic.evaluation.custom_coco_eval import CustomCOCOEvaluator
from detic.modeling.utils import reset_cls_test

# import SMNET samplers
from torch.utils.data import DistributedSampler
from SMNet.loader import SMNetDetectionLoader, collate_smnet
import yaml
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from torch.utils.data import DataLoader
from detectron2.structures import Instances, Boxes
import cv2
import numpy as np
import copy

# evaluation imports
# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
from torch import nn
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators
from pycocotools.coco import COCO
import json
# from torchviz import make_dot

# ROOT_DIR = '/home/nicolas/hpc-home/'
# ROOT_DIR = '/home/n11223243/'
ROOT_DIR = os.getcwd().split('allocentric_memory')[0]

# register new datasets
from detectron2.data.datasets import register_coco_instances
register_coco_instances("bdd_night", {}, f"{ROOT_DIR}ssod/dataset/C2N/test_data.json", f"{ROOT_DIR}ssod/dataset/C2N/test_data/")
register_coco_instances("bdd_daytime", {}, f"{ROOT_DIR}ssod/dataset/C2B/test_data.json", f"{ROOT_DIR}ssod/dataset/C2B/test_data/")
register_coco_instances("interactron_test", {}, f"{ROOT_DIR}interactron/data/interactron/test/coco_annotations.json", f"{ROOT_DIR}interactron/data/interactron/test/")
register_coco_instances("interactron_train", {}, f"{ROOT_DIR}interactron/data/interactron/train/coco_annotations.json", f"{ROOT_DIR}interactron/data/interactron/train/")

# mpd3 datasets
register_coco_instances("mp3d_val", {}, f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/val/annotations_reduced.json", f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/val")
# register_coco_instances("mp3d_val", {}, f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/val/mp3d_val.json", f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/val/JPEGImages")
register_coco_instances("mp3d_val_lvis", {}, f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/val/annotations_lvis.json", f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/val")
register_coco_instances("mp3d_train", {}, f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/train/annotations_reduced.json", f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/train")
register_coco_instances("mp3d_train_lvis", {}, f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/train/annotations_lvis.json", f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/train")
register_coco_instances("mp3d_test", {}, f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/test/annotations_reduced.json", f"{ROOT_DIR}allocentric_memory/Matterport/processed_data/test")

logger = logging.getLogger("detectron2")

def mp3d_inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(data_loader.dataset.__len__()))

    total = data_loader.dataset.__len__() # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)

    # add class metadata so that we get the class AP
    evaluator._metadata.thing_classes = [x['name'] for x in evaluator._coco_api.dataset["categories"]]

    # reset the evaluator
    evaluator.reset()
    scene_ids = []
    # second_half_ids = []
    # first_half_ids = []
    # first_quintile_ids = []
    # last_quintile_ids = []

    first_quintile_ids = []
    second_quintile_ids = []
    third_quintile_ids = []
    forth_quintile_ids = []

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        
        # build up json file with coco annotations
        im_id = 0
        det_id = 0
        new_annos = {"images": [], "annotations": []}
        for idx, inputs in enumerate(data_loader):
            # map to coco format
            inputs = map_mp3d_batch_to_coco(inputs)

            inputs = inputs[0]

            print(inputs[0]['image'].shape)
            print(inputs[0]['proj_indices'].shape)
            print(inputs[0]['memory'].shape)
            print(inputs[0]['memory_reset'])
            print(inputs[0]['observations'])

            # reset the scene_specific memory
            if inputs[0]['memory_reset']:
            # if idx%5==0:
                # if len(scene_ids) > 0:
                #     new_annos["categories"] = evaluator._coco_api.dataset["categories"]
                #     evaluator._coco_api.dataset = new_annos
                #     evaluator._coco_api.createIndex()
                #     results = evaluator.evaluate(scene_ids)
                #     logger.info("Evaluation results for {} in csv format:".format(current_scene))
                #     print_csv_format(results)

                # reset the scene evaluator
                scene_ids = []
                current_scene = inputs[0]['sequence_name'][0:13]

            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model([inputs])
            outputs = [outputs[i] for i in range(0, len(outputs), 5)]
            inputs = [inputs[i] for i in range(0, len(inputs), 5)]

            # add to annotations file
            for k in range(len(inputs)):
                # append to scene_ids
                scene_ids.append(im_id)

                # check if the image is in the second half of the sequence
                # if idx%50 > 25:
                #     second_half_ids.append(im_id)
                # else:
                #     first_half_ids.append(im_id)

                # UPDATE THE IDS FOR ASSESSING THE LONG-TERM RESULTS - IMAGE ONLY
                # if idx%50 < 25:
                #     first_quintile_ids.append(im_id)
                #     third_quintile_ids.append(im_id)
                # elif idx%50 < 50:
                #     second_quintile_ids.append(im_id)
                #     forth_quintile_ids.append(im_id)
                
                # UPDATE THE IDS FOR ASSESSING THE LONG-TERM RESULTS - MEMORY MODELS
                if idx%100 < 25:
                    first_quintile_ids.append(im_id)
                elif idx%100 < 50:
                    second_quintile_ids.append(im_id)
                elif idx%100 < 75:
                    third_quintile_ids.append(im_id)
                elif idx%100 >= 75:
                    forth_quintile_ids.append(im_id)

                # check if the image is in each quarter of the dataset
                # if idx%100 < 25:
                #     first_quintile_ids.append(im_id)
                # elif idx%100 < 50:
                #     second_quintile_ids.append(im_id)
                # elif idx%100 < 75:
                #     third_quintile_ids.append(im_id)
                # elif idx%100 >= 75:
                #     forth_quintile_ids.append(im_id)

                inputs[k]['image_id'] = im_id
                
                # add image to json file
                new_annos['images'].append({"id": inputs[k]['image_id'], "file_name": inputs[k]['file_name'], "height": inputs[k]['height'], "width": inputs[k]["width"]})

                # add annotations to json file
                for j in range(len(inputs[k]['instances'].gt_boxes)):
                    bbox = inputs[k]['instances'].gt_boxes.tensor[j]
                    new_annos['annotations'].append({"id": det_id, "image_id": im_id, "category_id": int(inputs[k]['instances'].gt_classes[j]), "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])], "iscrowd": 0, "area": 0})
                    det_id+=1
                im_id += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    # update evaluator to use the built annotations
    new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    # # save to output file
    # with open('{ROOT_DIR}allocentric_memory/Matterport/processed_data/val/mp3d_val.json', 'w') as fp:
    #     json.dump(new_annos, fp)

    # print out the final scene performance
    # new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    # evaluator._coco_api.dataset = new_annos
    # evaluator._coco_api.createIndex()
    # results = evaluator.evaluate(scene_ids)
    # logger.info("Evaluation results for {} in csv format:".format(current_scene))
    # print_csv_format(results)

    # print out the performance of episodes in the first quintile
    new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    evaluator._coco_api.dataset = new_annos
    evaluator._coco_api.createIndex()
    results = evaluator.evaluate(first_quintile_ids)
    logger.info("Evaluation results for second half episodes in csv format:")
    print_csv_format(results)

    # print out the performance of episodes in the second quintile
    new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    evaluator._coco_api.dataset = new_annos
    evaluator._coco_api.createIndex()
    results = evaluator.evaluate(second_quintile_ids)
    logger.info("Evaluation results for second half episodes in csv format:")
    print_csv_format(results)

    # print out the performance of episodes in the third quintile
    new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    evaluator._coco_api.dataset = new_annos
    evaluator._coco_api.createIndex()
    results = evaluator.evaluate(third_quintile_ids)
    logger.info("Evaluation results for second half episodes in csv format:")
    print_csv_format(results)

    # print out the performance of episodes in the forth quintile
    new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    evaluator._coco_api.dataset = new_annos
    evaluator._coco_api.createIndex()
    results = evaluator.evaluate(forth_quintile_ids)
    logger.info("Evaluation results for second half episodes in csv format:")
    print_csv_format(results)

    # print out the performance of episodes in the first half
    # new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    # evaluator._coco_api.dataset = new_annos
    # evaluator._coco_api.createIndex()
    # results = evaluator.evaluate(first_half_ids)
    # logger.info("Evaluation results for second half episodes in csv format:")
    # print_csv_format(results)

    # # print out the performance of episodes in the second half
    # new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    # evaluator._coco_api.dataset = new_annos
    # evaluator._coco_api.createIndex()
    # results = evaluator.evaluate(second_half_ids)
    # logger.info("Evaluation results for second half episodes in csv format:")
    # print_csv_format(results)

    # print out the performance of episodes in the final quintile
    # new_annos["categories"] = evaluator._coco_api.dataset["categories"]
    # evaluator._coco_api.dataset = new_annos
    # evaluator._coco_api.createIndex()
    # results = evaluator.evaluate(last_quintile_ids)
    # logger.info("Evaluation results for second half episodes in csv format:")
    # print_csv_format(results)
        
    evaluator._coco_api.dataset = new_annos
    evaluator._coco_api.createIndex()
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        if cfg.MODEL.RESET_CLS_TESTS:
            reset_cls_test(
                model,
                cfg.MODEL.TEST_CLASSIFIERS[d],
                cfg.MODEL.TEST_NUM_CLASSES[d])
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))

        # build mp3d sequence dataloader
        if cfg.DATALOADER.SAMPLER_TRAIN == 'MP3DLoader':
            # load the SMNet config
            with open(cfg.DATASETS.SMNET_CFG) as fp:
                cfg_smnet = yaml.safe_load(fp)
            world_size = 1

            # Setup Dataloader
            clip_path = cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH if cfg.MODEL.MEMORY_TYPE in ['semantic_gt', 'map_gt'] else None
            # semantic_gt = True if cfg.MODEL.MEMORY_TYPE == 'semantic_gt' else False
            memory_type = cfg.MODEL.MEMORY_TYPE
            semmap_path = '' # cfg.MODEL.SEMMAP_PATH
            v_loader = SMNetDetectionLoader(cfg_smnet["data"], split=cfg_smnet['data']['val_split'], memory_path=cfg.MODEL.MEMORY_PATH, test_type=cfg.MODEL.TEST_TYPE, clip_path = clip_path, memory_type=memory_type, semmap_path=semmap_path)
            v_sampler = InferenceSampler(len(v_loader))

            data_loader = DataLoader(
                v_loader,
                batch_size=1 // world_size,
                num_workers=cfg_smnet["training"]["n_workers"],
                drop_last=True,
                pin_memory=True,
                sampler=v_sampler,
                multiprocessing_context='fork',
                collate_fn=collate_smnet,
            )
        else:
            # build standard detection dataloader
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis" or cfg.GEN_PSEDO_LABELS:
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            if dataset_name == 'coco_generalized_zeroshot_val':
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)

        elif evaluator_type == 'oid':
            evaluator = OIDEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type

        # only evaluate bbox
        evaluator._tasks = ('bbox',)
        if cfg.DATALOADER.SAMPLER_TRAIN == 'MP3DLoader':
            results[dataset_name] = mp3d_inference_on_dataset(
                model, data_loader, evaluator)
        else:
            results[dataset_name] = inference_on_dataset(
                model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def map_mp3d_batch_to_coco(data):
    # iterate through samples in batch
    detection_batch = []
    for s in range(len(data)):
        detection_sequence = []
        for i in range(len(data[s])):
            file_name = data[s][i]['file_name']
            sequence_name = data[s][i]['sequence_name']
            gt_boxes = data[s][i]['gt_boxes']
            gt_classes = data[s][i]['gt_classes']
            rgb_image = data[s][i]['image']
            memory = data[s][i]['memory_features']
            proj_indices = data[s][i]['proj_indices']
            memory_reset = data[s][i]['memory_reset']
            observations = data[s][i]['observations']

            # print('File names: {}'.format(file_name))
            # print('GT boxes: {}'.format(gt_boxes))
            # print('GT classes: {}'.format(gt_classes))
            # print('RGB images: {}'.format(rgb_images))

            # create the output detection data
            detection_data = {}
            detection_data['file_name'] = file_name
            detection_data['sequence_name'] = sequence_name
            detection_data['height'] = rgb_image.shape[0]
            detection_data['width'] = rgb_image.shape[1]

            # create the instances object
            instances = Instances(image_size=(rgb_image.shape[0], rgb_image.shape[1]))
            instances.set('gt_boxes', Boxes(torch.tensor(gt_boxes)))
            instances.set('gt_classes', torch.tensor(gt_classes))
            detection_data['instances'] = instances

            # apply image transformations
            rgb_image = torch.tensor(rgb_image)
            # rgb_image = rgb_image[:, :, [2, 1, 0]] # swap channels
            rgb_image = rgb_image.permute(2, 0, 1) 
            detection_data['image'] = rgb_image

            # add memory to the detection data
            detection_data['memory'] = memory
            detection_data['proj_indices'] = proj_indices
            detection_data['memory_reset'] = memory_reset
            detection_data['observations'] = observations
        
            # add to sequences
            detection_sequence.append(detection_data)

        # append to the batch
        detection_batch.append(detection_sequence)
            
    # update data
    data = detection_batch

    return data

def do_train(cfg, model, resume=False):
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    use_custom_mapper = cfg.WITH_IMAGE_LABELS
    MapperClass = CustomDatasetMapper if use_custom_mapper else DatasetMapper
    mapper = MapperClass(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DetrDatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == 'DETR' else \
        MapperClass(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    elif cfg.DATALOADER.SAMPLER_TRAIN == 'MP3DLoader':
        # load the SMNet config
        with open(cfg.DATASETS.SMNET_CFG) as fp:
            cfg_smnet = yaml.safe_load(fp)
        # world size = 1
        world_size = 1
        # Setup Dataloader
        clip_path = cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH if cfg.MODEL.MEMORY_TYPE in ['semantic_gt', 'map_gt'] else None
        # semantic_gt = True if cfg.MODEL.MEMORY_TYPE == 'semantic_gt' else False
        memory_type = cfg.MODEL.MEMORY_TYPE
        semmap_path = cfg.MODEL.SEMMAP_PATH
        t_loader = SMNetDetectionLoader(cfg_smnet["data"], split=cfg_smnet['data']['train_split'], memory_path=cfg.MODEL.MEMORY_PATH, clip_path = clip_path, memory_type=memory_type, semmap_path=semmap_path)
        t_sampler = TrainingSampler(len(t_loader))

        # t_sampler = DistributedSampler(t_loader)

        data_loader = DataLoader(
            t_loader,
            batch_size=cfg.SOLVER.IMS_PER_BATCH // world_size,
            num_workers=cfg_smnet["training"]["n_workers"],
            drop_last=True,
            pin_memory=True,
            sampler=t_sampler,
            multiprocessing_context='fork',
            collate_fn=collate_smnet,
        )
    else:
        data_loader = build_custom_train_loader(cfg, mapper=mapper)

    if cfg.FP16:
        scaler = GradScaler()

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # map the mp3d batch to coco format
            data = map_mp3d_batch_to_coco(data)

            # for j in range(len(data)):
            #     # check the dataset
            #     image = data[j]["image"].numpy()
            #     m = image.transpose((1, 2, 0)).astype(np.uint8).copy() 
            #     print(data[j])
            #     # draw bboxes on image
            #     for i in range(len(data[j]["instances"].gt_boxes)):
            #         bbox = data[j]["instances"].gt_boxes.tensor[i]
            #         image = cv2.rectangle(m, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            #     cv2.imshow('image', m)
            #     cv2.waitKey(0)
            # cv2.destroyAllWindows()

            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)


            # print_params = {}
            # for k, v in model.named_parameters():
            #     if 'map_merge' in k:
            #         print_params[k] = v
            # make_dot(loss_dict['loss_cls_stage0'], params=print_params).render("attached", format="png")

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if cfg.FP16:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)

    # add new config elements related to semantic mapping
    # cfg.DATASETS.SMNET_CFG = "SMNet/smnet.yml"
    # cfg.MODEL.MAP_MERGE_TYPE = ''
    # cfg.MODEL.MAP_FEAT_FUSION = ''
    # cfg.MODEL.FREEZE_BACKBONE = False
    # cfg.MODEL.UNFROZEN_LAYERS = []

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))

        # add merge type to the output file name
        if cfg.MODEL.MAP_MERGE_TYPE:
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '_{}'.format(cfg.MODEL.MAP_MERGE_TYPE) + '_{}'.format(cfg.MODEL.MEMORY_TYPE) + '_{}'.format(cfg.MODEL.MEMORY_FEATURE_WEIGHT) + \
                '_{}'.format(cfg.MODEL.MAP_FEATURE_WEIGHT) + '_{}'.format(cfg.MODEL.MEMORY_CLS_SCORE_THRESH) # + '_{}'.format(cfg.MODEL.MEMORY_OBS_SCORE_THRESH)
        # add the date as a subfolder
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="detic")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    
    # freeze the backbone layers, keeping the detection head and merging layers trainable
    if cfg.MODEL.FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            for layer in cfg.MODEL.UNFROZEN_LAYERS:
                if layer in name:
                    param.requires_grad = True
                    break  
                param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return do_test(cfg, model)

    if args.save_embeddings:
        # make directory for region embeddings
        default_dir = 'prompt_learning/temp/'

        # check if directory exists
        if not os.path.exists(default_dir):
            os.makedirs(default_dir + 'embeddings/')
            os.makedirs(default_dir + 'proposals/')
            os.makedirs(default_dir + 'scores/')
            os.makedirs(default_dir + 'objectness/')


        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return_test = do_test(cfg, model)

        # embedding_dir = 'prompt_learning/embeddings/' + cfg.DATASETS.TEST[0]
        # # rename default_dir to embedding_dir
        # os.rename(default_dir, embedding_dir)

        return return_test

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)

def argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--save-embeddings", action="store_true", help="save roi embeddings")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    
    # add new dataset
    parser.add_argument("--custom-annos", type=str, default="")
    parser.add_argument("--custom-imgs", type=str, default="")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        if args.dist_url == 'host':
            args.dist_url = 'tcp://{}:12345'.format(
                os.environ['SLURM_JOB_NODELIST'])
        elif not args.dist_url.startswith('tcp'):
            tmp = os.popen(
                    'echo $(scontrol show job {} | grep BatchHost)'.format(
                        args.dist_url)
                ).read()
            tmp = tmp[tmp.find('=') + 1: -1]
            args.dist_url = 'tcp://{}:12345'.format(tmp)

    if args.custom_annos:
        print('here')
        register_coco_instances("custom_dataset", {}, args.custom_annos, args.custom_imgs)


    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
