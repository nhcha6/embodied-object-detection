# allocentric_memory
## Data Preparation ##
1) cd Semantic-Mapnet
2) Run create_coco_replica.py to create coco annotations and JPEGImages
3) Run precompute_training_inputs/build_replica_data.py to generate the core sequence data used to train and test Semantic-Mapnet
<!-- 4) Run build_smnet_features.py to run SMNet inference and save the spatial memory tensors. Due to memory, we run each method on the first 500 images in the sequence and save the resulting representation to file. Again, due to memory constraints, some
   sequences did not finish. We need to move towards running SMNet recurrently such that the entire sequenece does not need to be stored in memory to overcome these limitations. -->
5) Run precompute_training_inputs/build_replica_memory_features.py to generate the compressed spatial memory and projection indices required by the dataloader. Needs to be run separately for the continuous testing sequences.
7) Copy across replica_map_info.json to have map dimension information for visualisation
