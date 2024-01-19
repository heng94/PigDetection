import torch
import torch.nn as nn
from .clip import clip


class TextEncoder(nn.Module):
    
    def __init__(self, cfg):
        
        super(TextEncoder, self).__init__()
        self.cfg = cfg
        self.tokenizer = clip.tokenize
        self.model, _ = clip.load(cfg.clip_pretrain, device=cfg.device)
        
        # freeze the parameters
        if getattr(self.model, 'module', False):
            for child in self.model.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text_input):
        
        with torch.no_grad():
            text = self.tokenizer(text_input).to(self.cfg.device)
            text_features = self.model.encode_text(text)
            
            return text_features