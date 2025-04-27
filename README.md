# Enhanced Embodied Object Detection with Spatial Feature Memory #
Official repository for the paper [Enhanced Embodied Object Detection with Spatial Feature Memory](https://ieeexplore.ieee.org/abstract/document/10944108). This repository contains code for:
* Performing inference with the proposed embodied object detectors on an [example dataset](#example-data-preparation)
* Performing infrerence the proposed embodied object detectors on data collected by a [real robot](#demo-on-real-robot)
* [Generating the complete dataset](#generate-the-full-dataset-for-training-and-testing) used for testing and training
* Training the proposed [embodied object detector](#train-models)

## Example data preparation ##
We provide a small example dataset for testing the models trained in our paper. This is NOT the complete dataset, but enables developers to familiarise themselves with the method without having to generate the complete the dataset.
1) Download example data from [here](https://1drv.ms/u/s!AnUcX0micjmciuR7FYs7_4i9bKK6PA) and place in Detic/embodied_data/
2) Download the models used in our experiments from [here](https://1drv.ms/u/s!AnUcX0micjmciuU3ozLp9aLgWJ1NwQ?e=fQFWBV) here and place in Detic/models
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
Prepare the virtual environment using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#mamba-install) to run the embodied object detectors on Linux systems
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
1) Detic Pre-trained

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

## Generate the full dataset for training and testing
First, we need to create a new environment for running the habitat simulator, separate to that used to test and train the embodied object detector. 
1) Create the new environment
	```bash
	mamba create --name habitat_env python=3.6
	mamba activate habitat_env
	```
2) Install the required pytorch and opencv versions
	```bash
	mamba install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -c pytorch -c nvidia
	pip install opencv-python==3.4.0.14
	```
Habitat-sim version [v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) and habitat-lab [v0.1.5](https://github.com/facebookresearch/habitat-lab/tree/v0.1.5) then need to be install. We provide instructions for installing these packages on linux using mamba. We further include in this repo a version of habitat-lab with minor changes to ensure compatability with both the Replica and Matterport datasets.
1) Install habitat-sim
	```bash
	mamba install habitat-sim=0.1.7 -c conda-forge -c aihabitat
	```
 2) Install requirements for running habitat
	```bash
	mamba install -c conda-forge scikit-build pyyaml h5py yacs
 	pip install torch-scatter==1.4.0
	```
 3) Navigate to the provided habitat-lab source code and build the package
	```bash
	cd embodied-object-detection/habitat-lab
	python setup.py develop --all
	```
 
Next, we need to prepare the Matterport3D and Replica datasets for performing embodied object detetection:
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
	python build_data.py --data_path /user/path/to/Matterport/
	```
 	```bash
	python build_replica_data.py --data_path /user/path/to/Replica-Dataset/
	``` 
5) Runs scripts to generate the projection infomration used to read and write to external memory
	```bash
	python build_memory_data.py --data_path /user/path/to/Matterport/
	```
 	```bash
	python build_replica_memory_data.py --data_path /user/path/to/Replica-Dataset/
	```  

## Train models
The first step in our training pipeline is to fine-tune Detic using the image data only. To fine-tune Detic on the training set while using the validation dataset for testing, run the following:

	python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth MODEL.MEMORY_TYPE image_only MODEL.TRAIN_DATA_PATH embodied_data/mp3d_train/ MODEL.TEST_DATA_PATH embodied_data/mp3d_val/ OUTPUT_DIR output/finetuned_detic_train/

Note the path of the model with the best performance on the validation dataset. This model should be used for testing, and also to precompute the implicit object memory for training our proposed embodied object detector. Run the following to calculate the implicit object memory, ensuring the model weights correspond to your fine-tuned Detic model.

	python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/detic_finetuned.pth MODEL.TEST_DATA_PATH embodied_data/mp3d_train/ OUTPUT_DIR output/finetuned_detic_inference/ MODEL.TEST_SAVE_SEMMAP True

We can now use the precomputed memory to train the proposed embodied object detector. To do so, run 

	python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml MODEL.WEIGHTS models/detic_finetuned.pth MODEL.TRAIN_DATA_PATH embodied_data/mp3d_train/ MODEL.TEST_DATA_PATH embodied_data/mp3d_val/ OUTPUT_DIR output/implicit_object_memory_train/ MODEL.SEMMAP_PATH output/finetuned_detic_inference/memory

