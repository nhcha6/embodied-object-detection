cd ..

#### EVALUATE PRETRAINED MODEL ON MP3D DATASET USING SMNET DATALOADER ####
# pretrained model
# python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_custom_dataloader.yaml --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth

# novel backbone
python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_map_conditioned.yaml --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth MODEL.MAP_FEAT_FUSION sum

#### FINETUNE MODEL ON MP3D DATASET USING SMNET DATALOADER ####
# pretrained model
# python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_custom_dataloader.yaml MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth

# novel backbone
# python train_mp3d.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_map_conditioned.yaml MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth

