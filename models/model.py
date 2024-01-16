import torch 
import torch.nn as nn
from visual_encoder import visual_encoder, MultiScaleFusion
from text_encoder import TextEncoder
from detection_head import WeakREChead
from utils import get_boxes


class PigDetector(nn.Module):
    
    def __init__(self, cfg):
        
        super(PigDetector, self).__init__()
        self.cfg = cfg
        self.visual_encoder = visual_encoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        self.linear_vs = nn.Linear(1024, cfg.hidden_size)
        self.linear_tx = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.head = WeakREChead()
        
    def forward(self, image_input, text_input):
        
        _, output, boxes_sml = self.visual_encoder(image_input)
        text_features = self.text_encoder(text_input)
        
        # Vision Multi Scale Fusion
        small_out, middle_out, large_out = output
        feature_input = [large_out, middle_out, small_out]
        new_large, new_middle, new_small = self.multi_scale_manner(feature_input)
        new_features = [new_large, new_middle, new_small]
        
        # Anchor Selection
        new_boxes_sml = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True).squeeze(2)[:, :, 4]
        _, indices = mean_i.topk(k=int(self.cfg.select_num), dim=1, largest=True, sorted=True)
        batch_size, grid_num, anchor_num, chal_num = boxes_sml[0].shape
        select_num = indices.shape[1]
        new_box_sml = boxes_sml[0].masked_select(torch.zeros(batch_size, grid_num))
        new_box_sml = new_box_sml.to(boxes_sml[0].device).scatter(1, indices, 1).bool()
        new_box_sml = new_box_sml.unsqueeze(2).unsqueeze(3).expand(batch_size, grid_num, anchor_num, chal_num)
        new_box_sml = new_box_sml.contiguous().view(batch_size, select_num, anchor_num, chal_num)
        new_boxes_sml.append(new_box_sml)
        
        batch_size, dim_num, h, w = new_features[0].size()
        new_feature = new_features[0].view(batch_size, dim_num, h * w).permute(0, 2, 1)
        _, grid_num, chal_num = new_feature.shape
        new_feature = new_feature.masked_select(torch.zeros(batch_size, grid_num).to(new_feature.device))
        new_feature = new_feature.scatter(1, indices, 1).bool().unsqueeze(2).expand(batch_size, grid_num, chal_num)
        new_feature = new_feature.contiguous().view(batch_size, select_num, chal_num)
        
        linear_feature = self.linear_vs(new_feature)
        linear_text = self.linear_tx(text_features)
        
        if self.training:
            loss = self.head(linear_feature, linear_text)
            
            return loss
        else:
            predictions_s = self.head(linear_feature, linear_text)
            predictions_list = [predictions_s]
            box_pred = get_boxes(new_boxes_sml, predictions_list, self.cfg.class_num)
            
            return box_pred
        