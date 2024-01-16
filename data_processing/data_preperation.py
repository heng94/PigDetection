import os 
import shutil
import numpy as np

data_root = '/ZhouHeng/Data/PigTracking/pig_test_4'
os.makedirs('./small_dataset', exist_ok=True)
os.makedirs('./small_dataset/images', exist_ok=True)
splits = ['train', 'test']

# open a txt file to save the data path
txt_file = open('./small_dataset/data_path_v1.txt', 'w')

# list all sub_dirs in root dir
for split in splits:
    dirs = os.listdir(os.path.join(data_root, split))
    for sub_dir in dirs:
        print(sub_dir)
        os.makedirs(f'./small_dataset/images/{sub_dir}', exist_ok=True)
        file_path = os.path.join(data_root, split, sub_dir, 'img1')
        file_names = os.listdir(file_path)
        num_files = len(file_names)
        random_range = np.arange(num_files).tolist()
        
        # random select 100 images from each sub_dir
        if num_files > 100:
            idx = np.random.choice(random_range, 100, replace=False).tolist()
            idx.sort()
            for i in idx:
                shutil.copy(os.path.join(file_path, file_names[i]), f'./small_dataset/images/{sub_dir}')
                txt_file.write(os.path.join(f'./images/{sub_dir}', file_names[i]) + '\n')
        else:
            for i in range(num_files):
                shutil.copy(os.path.join(file_path, file_names[i]), f'./small_dataset/images/{sub_dir}')
                txt_file.write(os.path.join(f'./images/{sub_dir}', file_names[i]) + '\n')
                