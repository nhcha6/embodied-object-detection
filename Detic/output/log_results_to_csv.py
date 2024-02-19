import numpy as np
import plotly.graph_objects as go

# files = ['output/Detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent_map_gt_sum/2023-11-22_10-11-24/log.txt',
#          'output/Detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent_implicit_memory_sum/2023-11-22_10-12-20/log.txt',
#          'output/Detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent_implicit_memory_sum/2023-11-22_10-57-13/log.txt',
#          'output/Detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent_implicit_memory_sum/2023-11-22_17-39-20/log.txt',
#          'output/Detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size_mp3d_recurrent_implicit_memory_sum/2023-11-22_17-45-14/log.txt']

# names = ['explicit_map memory initialised from pixel memory',
#          'pixel memory and explicit map memory initialised from pixel memory',
#          'pixel memory and explicit map memory initialised from explicit map memory',
#          'pixel memory initialised from explicit map, but with backbone frozen',
#          'pixel memory and explicit map memory initialised from explicit map memory, but with backbone frozen']

# files = [
#     'output/mp3d_16_long_short/explicit_map memory initialised from pixel memory/log.txt',
#     'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from pixel memory/log.txt',
#     'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from pixel memory run 2/log.txt',
#     'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from explicit map memory, but with backbone frozen/log.txt',
#     'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from explicit map memory, but with backbone frozen run 2/log.txt'
# ]

files = [
    'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from explicit map memory, but with backbone frozen/5999_test_val.txt',
    'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from explicit map memory, but with backbone frozen run 2/train_val_4999.txt',
    'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from pixel memory/test_val_6999.txt',
    'output/mp3d_16_long_short/pixel memory and explicit map memory initialised from pixel memory run 2/test_val_6999.txt'
]



for j, file in enumerate(files):
    # open the two log files
    file1 = open(file, 'r').readlines()

    # find the index of each line that contains AP50
    index1 = []
    for i, line in enumerate(file1):
        if 'AP50' in line:
            index1.append(i+1)

    index1 = index1[1::2]

    # print the lines that contain AP50
    results = [file1[i].split(':')[4].split(',')[0:3] for i in index1]
    # convert to float
    results = [[float(x) for x in y] for y in results]
    
    # for the test results
    scene_results = np.array(results[0:-5])
    overall_results = np.array(results[-5:])

    # for the validation results
    validation_results = results[9::10]
    validation_results_2 = results[11::12]

    # print the training results
    # print(file)
    # if j<2:
    #     print(validation_results)
    # else:
    #     print(validation_results_2)

    # print the test results
    print(file)
    print(overall_results)



