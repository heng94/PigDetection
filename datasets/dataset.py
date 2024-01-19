import os
import json
import torch
import numpy as np
from PIL import Image
from types import Any
from utils import label2yolobox
from torchvision import transforms
from torch.utils.data import Dataset


class PigCOCODataset(Dataset):
    
    def __init__(self, cfg: str, split: str) -> None:
        super(PigCOCODataset, self).__init__()
        self.cfg = cfg
        self.split = split
        state_refs_list = json.load(open(cfg.anno_path[cfg.dataset], 'r'))
        self.refs_anno = state_refs_list[self.split]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(self.cfg.mean, self.cfg.mean.std)
        ])
        
    def load_image(self, idx: int) -> Any:
        
        image_id = self.refs_anno[idx]['iid']
        img_path = os.path.join(self.cfg.data_root, self.cfg.dataset, f'{image_id}.jpg')
        image = Image.open(img_path)
        
        return image, image_id
    
    def load_bbox(self, idx: int) -> Any:
        
        bbox = np.array([self.refs_anno[idx]['bbox']])
        
        return bbox
    
    def preprocessing(self, image: Any, bbox: Any, image_id: int, lr_flip: bool=False) -> Any:
        
        h, w, _ = image.shape
        img_size = self.cfg.input_shape
        new_ar = w / h
        if new_ar < 1:
            new_h = img_size
            new_w = new_h * new_ar
        else:
            new_w = img_size
            new_h = new_w / new_ar
        
        new_w, new_h = int(new_w), int(new_h)
        dx = (img_size - new_w) // 2
        dy = (img_size - new_h) // 2
        
        new_img = image.resize(new_w, new_h)
        
        sized = np.ones((img_size, img_size, 3), dtype=np.uint8) * 127
        sized[dy: dy + new_h, dx: dx + new_w, :] = new_img
        img_info = (h, w, new_h, new_w, dx, dy, image_id)
        sized_bbox = label2yolobox(bbox, img_info, self.input_shape[0], lrflip=lr_flip)
        
        return sized, sized_bbox, img_info
        
    def __getitem__(self, idx) -> Any:
        
        prompt = self.refs_anno[idx]['refs']
        image, image_id = self.load_image(idx)
        gt_bbox = self.load_bbox(idx)
        image, bbox, img_info = self.preprocessing(image, gt_bbox.copy(), image_id)
        
        image = self.transform(image)
        bbox = torch.from_numpy(bbox).float()
        gt_bbox = torch.from_numpy(gt_bbox).float()
        img_info = np.array(img_info)
        
        if self.split == 'train':
            return prompt, image
        else:
            return prompt, image, bbox, gt_bbox, img_info
        
    def __len__(self) -> int:
        
        return len(self.refs_anno)