# coding=utf-8
import torch
import torch.nn as nn


def get_contrast(vis_emb, lan_emb):
    
    sim_map = torch.einsum('avd, bqd -> baqv', vis_emb, lan_emb)
    batch_size = sim_map.shape[0]
    max_sims, _ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True)
    max_sims = max_sims.squeeze(2)

    # Negative Anchor Augmentation
    max_sim_0, max_sim_1 = max_sims[...,0],max_sims[...,1]
    max_sim_1 = max_sim_1.masked_select(
        ~torch.eye(batch_size).bool().to(max_sim_1.device)
    ).contiguous().view(batch_size, batch_size-1)
    new_logits = torch.cat([max_sim_0, max_sim_1], dim=1)

    target = torch.eye(batch_size).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    
    return loss

def get_prediction(vis_emb, lan_emb):
    sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    _, v = sim_map.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2, v.expand(sim_map.shape), 1).bool()
    
    return predictions


class WeakREChead(nn.Module):
    
    def __init__(self, ):
        
        super(WeakREChead, self).__init__()

    def forward(self, vis_fs,lan_fs):
        
        if self.training:
            loss = get_contrast(vis_fs, lan_fs)
            
            return loss
        else:
            predictions = get_prediction(vis_fs, lan_fs)
            
            return predictions










