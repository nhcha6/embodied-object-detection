# Enhanced Embodied Object Detection with Language Image Pre-training and Implicit Object Memory
## Example Data Preparation ##
1) Download example data and place in Detic/embodied_data/
2) Download pre-trained DETIC models and place in Detic/models
	- ImageNet Backbone: https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth
	- DETIC: https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
3) Download the models trained in our experiments from here and place in models

## Running Embodied Object Detection with Implicit Object Memory ##
1) Detic Pretrained:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth MODEL.MEMORY_TYPE image_only MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/pre-trained/

2) Vanilla Training: 

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/vanilla_training.pth MODEL.MEMORY_TYPE image_only MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/vanilla/

3) DETIC Fine-tuned:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/detic_finetuned.pth MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/finetuned_detic/

4) DETIC Pre-trained:

		python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent.yaml --eval-only MODEL.WEIGHTS models/implicit_object_memory.pth MODEL.MAP_FEAT_FUSION sum MODEL.MEMORY_TYPE implicit_memory MODEL.MAP_FEATURE_WEIGHT 5 MODEL.TEST_DATA_PATH embodied_data/mp3d_example/ OUTPUT_DIR output/implicit_object_memory/

## Generate the full dataset for training/testing
1) cd Semantic-Mapnet
2) Run create_coco_replica.py to create coco annotations and JPEGImages
3) Run precompute_training_inputs/build_replica_data.py to generate the core sequence data used to train and test Semantic-Mapnet
<!-- 4) Run build_smnet_features.py to run SMNet inference and save the spatial memory tensors. Due to memory, we run each method on the first 500 images in the sequence and save the resulting representation to file. Again, due to memory constraints, some
   sequences did not finish. We need to move towards running SMNet recurrently such that the entire sequenece does not need to be stored in memory to overcome these limitations. -->
5) Run precompute_training_inputs/build_replica_memory_features.py to generate the compressed spatial memory and projection indices required by the dataloader. Needs to be run separately for the continuous testing sequences.
7) Copy across replica_map_info.json to have map dimension information for visualisation

## Pre-compute memory data before training
