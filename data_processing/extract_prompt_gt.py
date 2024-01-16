import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import random
from PIL import Image
from lavis.models import load_model_and_preprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", 
    model_type="caption_coco_opt6.7b", 
    is_eval=True, 
    device=device
)

json_file = {}
anno_list = []
data_root = './small_dataset/images'
dir_list = os.listdir(data_root)

count = 0
for sub_dir in dir_list:
    sub_dir_path = os.path.join(data_root, sub_dir)
    image_list = os.listdir(sub_dir_path)
    
    for image_name in image_list:
        anno_dict = {}
        gt_file = open(os.path.join(sub_dir_path, 'gt.txt'), 'r')
        if image_name.endswith('.jpg'):
            image_path = os.path.join(sub_dir_path, image_name)
            print(f'Processing {image_path}')
            image = Image.open(image_path)
            # 0.jpg -> 000000.jpg
            new_name = str(count).zfill(6) + '.jpg'
            image.save(os.path.join(f'./small_dataset/data', new_name))
            image = vis_processors["eval"](image).unsqueeze(0).to(device)
            caption = model.generate({"image": image})
            anno_dict['iid'] = str(count).zfill(6)
            anno_dict['cat_id'] = 0
            anno_dict['refs'] = caption
            anno_dict['mask_id'] = count
            anno_dict['bbox'] = []
            # read each line in gt.txt
            for line in gt_file.readlines():
                line = line.strip().split(',')
                if int(line[0]) == int(image_name.split('.')[0]):
                    anno_dict['bbox'].append([float(line[2]), float(line[3]), float(line[4]), float(line[5])])
            gt_file.close()
            anno_list.append(anno_dict)
            count += 1

# shuffle anno_list
random.shuffle(anno_list)

# 80% of anno_list for train, 10% val and 10% test
train_list = anno_list[:int(len(anno_list)*0.8)]
val_list = anno_list[int(len(anno_list)*0.8):int(len(anno_list)*0.9)]
test_list = anno_list[int(len(anno_list)*0.9):]
json_file['train'] = train_list
json_file['val'] = val_list
json_file['test'] = test_list

# save json file
with open('./small_dataset/refpig_blip2_opt.json', 'w') as f:
    json.dump(json_file, f)
    
# print('1')

            
            
            

