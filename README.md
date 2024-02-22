# Enhanced Embodied Object Detection with Language Image Pre-training and Implicit Object Memory
## Example data preparation ##
1) Download example data from [here](https://1drv.ms/u/s!AnUcX0micjmciuR7uvg0ITSEiJh3Yg?e=vd708r)(https://1drv.ms/u/s!AnUcX0micjmciuR7FYs7_4i9bKK6PA?e=LvK5cO) and place in Detic/embodied_data/
2) Download the models used in our experiments from [here](https://1drv.ms/u/s!AnUcX0micjmciuR6rLJOb9RVjT5sgQ?e=Dg4wUn) here and place in Detic/models
3) The added data should have the following structure:
```bash
embodied-object-detection
└── Detic
    ├── embodied_data
    │   └── mp3d_example
    │       └── ...
    └── models
        ├── detic_finetuned.pth
        ├── Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth
        ├── implicit_object_memory.pth
        ├── resnet50_miil_21k.pth
        └── vanilla_training.pth
```

## Virtual environment preparation ##
Example virtual environment set-up on Linux:
```bash
# create environment and install pytorch
mamba create --name eod_env python=3.10 -y
mamba activate eod_env
mamba install pytorch=1.12 torchvision cudatoolkit=11.3 -c pytorch-lts -c nvidia

# install detectron2
cd embodied-object-detection
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .

# install remaining requirements
cd ../Detic
pip install -r requirements.txt
```

## Inference on example data ##
1) Detic Pre-trained:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth MODEL.MEMORY_TYPE image_only MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/pre-trained/

2) Vanilla Training: 

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/vanilla_training.pth MODEL.MEMORY_TYPE image_only MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/vanilla/

3) DETIC Fine-tuned:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/detic_finetuned.pth MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/finetuned_detic/

4) Fine-tuned DETIC + Implicit Object Memory:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/implicit_object_memory.pth MODEL.MAP_FEAT_FUSION sum MODEL.MEMORY_TYPE implicit_memory MODEL.MAP_FEATURE_WEIGHT 5 MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/implicit_object_memory/

## Demo on real robot
1) Download our collected data from [here](https://1drv.ms/u/s!AnUcX0micjmciuR8Uh3m3RejabJ2kg?e=3bd6ps) and place in Detic/embodied_data. The folder structure should look as follows:
```bash
embodied-object-detection
└── Detic
    ├── embodied_data
        ├── mp3d_example
        │   └── ...
        └── robot_example
            └── lap1
            └── lap2
            └── lapcw1
```
2) To run fine-tuned detic with implicit object memory:
    
		python robot_demo.py --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --vocabulary mp3d --confidence-threshold 0.3 --data_path embodied_data/robot_example/ --opts  MODEL.WEIGHTS models/implicit_object_memory.pth MODEL.MEMORY_TYPE implicit_memory MODEL.MAP_FEATURE_WEIGHT 5

3) To run other models, update MODEL.WEIGHTS and MODEL.MEMORY_TYPE as per the 'Inference on example data'

## Generate the full dataset for training/testing
Coming soon
<!-- 1) cd Semantic-Mapnet
2) Run create_coco_replica.py to create coco annotations and JPEGImages
3) Run precompute_training_inputs/build_replica_data.py to generate the core sequence data used to train and test Semantic-Mapnet
4) Run precompute_training_inputs/build_replica_memory_features.py to generate the compressed spatial memory and projection indices required by the dataloader. Needs to be run separately for the continuous testing sequences.
5) Copy across replica_map_info.json to have map dimension information for visualisation -->

## Train models
Coming soon

## ROS demo
Coming soon

