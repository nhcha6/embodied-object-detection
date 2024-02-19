cd ..

# evaluate pretrained model on dataset
# python train_net.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size_mp3d.yaml --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# python train_net.py --num-gpus 1 --config-file configs/Detic_DeformDETR_LI_R50_4x_ft4x_mp3d.yaml --eval-only MODEL.WEIGHTS models/Detic_DeformDETR_LI_R50_4x_ft4x.pth

python train_net.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d.yaml --eval-only MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth

# fine-tune pretrained model on dataset
# python train_net.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size_mp3d.yaml MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# python train_net.py --num-gpus 1 --config-file configs/Detic_DeformDETR_LI_R50_4x_ft4x_mp3d.yaml MODEL.WEIGHTS models/Detic_DeformDETR_LI_R50_4x_ft4x.pth

# python train_net.py --num-gpus 1 --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d.yaml MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth
