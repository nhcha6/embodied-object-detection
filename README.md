# Enhanced Embodied Object Detection with Language Image Pre-training and Implicit Object Memory
Official repository for the paper 'Enhanced Embodied Object Detection with Language Image Pre-training and Implicit Object Memory'.

## Example data preparation ##
1) Download example data from [here](https://1drv.ms/u/s!AnUcX0micjmciuR7FYs7_4i9bKK6PA) and place in Detic/embodied_data/
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
Example virtual environment set-up on Linux using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#mamba-install):
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
We provide scripts to calculate evaluation metrics for the pre-trained models on the example data.
1) Detic Pre-trained:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth MODEL.MEMORY_TYPE image_only MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/pre-trained/

2) Vanilla Training: 

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/vanilla_training.pth MODEL.MEMORY_TYPE image_only MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/vanilla/

3) DETIC Fine-tuned:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/detic_finetuned.pth MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/finetuned_detic/

4) Fine-tuned DETIC + Implicit Object Memory:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/implicit_object_memory.pth MODEL.MAP_FEAT_FUSION sum MODEL.MEMORY_TYPE implicit_memory MODEL.MAP_FEATURE_WEIGHT 5 MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/implicit_object_memory/

## Demo on real robot
We also provide the code and data to perform inference with the pre-trained models on data collected with a real robot. This script visualises the input to the embodied object detector, and overlays the output detections on the RGB image and the spatial memory.
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

3) To run other models, update MODEL.WEIGHTS and MODEL.MEMORY_TYPE as per the [Inference on example data](#inference-on-example-data)

## Generate the full dataset for training/testing
First, create a new virtual environment to run the habitat simulator:


To prepare the Matterport3D and Replica datasets for testing and training embodied object detetection:
1) Download the Matterport data from [here](https://niessner.github.io/Matterport/) and the Replica data from [here](https://github.com/facebookresearch/Replica-Dataset)
2) Navigate to the SMNet folder
	```bash
	cd SMNet
	```
3) Run scripts to generate a coco dataset from the Matterport and Replica scenes
	```bash
	python create_coco_mp3d.py --data_path /user/path/to/Matterport/
	```
 	```bash
	python create_coco_replica.py --data_path /user/path/to/Replica-Dataset/
	```
4) Runs scripts to generate the sensor data for embodied object detections from the Matterport and Replica scenes
	```bash
	python create_coco_mp3d.py --data_path /user/path/to/Matterport/
	```
 	```bash
	python create_coco_replica.py --data_path /user/path/to/Replica-Dataset/
	``` 
5) Runs scripts to generate the projection infomration used to read and write to external memory
	```bash
	python build_memory_data.py --data_path /user/path/to/Matterport/
	```
 	```bash
	python build_replica_memory_data.py --data_path /user/path/to/Replica-Dataset/
	```  

## Train models
Coming soon


