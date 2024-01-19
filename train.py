import os
import time
import torch
import wandb
import argparse
from datasets.dataset import PigCOCODataset
from models.model import PigDetector
from models.ema import EMA
from utils.config import load_config
from utils.utils import AverageMeter
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger


def train_one_epoch(fabric, model, optimizer, lr_scheduler, train_dataloader, cfg, epoch):
    
    # set model to train mode
    model.train()
    fabric.print(f'======> The model is training on Epoch {epoch}')
    
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    lr = AverageMeter('lr', ':.5f')
    
    meterics = [batch_time, data_time, losses, lr]
    meters_dict = {meteric.name: meteric for meteric in meterics}
    
    end = time.time()
    
    for batch_idx, (prompt, image) in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        
        is_accumulating = batch_idx % cfg.train.accumulation_steps != 0
        
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            with fabric.autocast(enabled=cfg.train.use_amp):
                loss = model(image, prompt)
                fabric.backward(loss)
                fabric.clip_gradients(model, optimizer, clip_val=cfg.train.clip_grad_norm)
        
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()
        
        if cfg.train.use_ema:
            model.update_params()
            
        losses.update(loss.item(), cfg.train.batch_size)
        lr.update(optimizer.param_groups[0]["lr"], -1)
        
        with fabric.rank_zero_first(local=False):
            global_step = epoch * len(train_dataloader) + batch_idx
            fabric.log_metrics(meters_dict, global_step=global_step)
            wandb.log({'Train loss': loss.avg_reduce, 'Train lr': lr.avg_reduce, 'Train Epoch': epoch})
        
        fabric.print(
            f'Epoch: [{epoch}][{batch_idx}/{len(train_dataloader)}]\t'
            f'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            f'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            f'Train loss {losses.val:.4f} ({losses.avg:.4f})\t'
            f'Train lr {lr.val:.5f} ({lr.avg:.5f})\t'
        )
        batch_time.update(time.time() - end)
        end = time.time()
        
        
@torch.no_grad()
def val_one_epoch(fabric, model, val_dataloader, cfg, epoch):
    
    if cfg.train.use_ema:
        model.apply_shadow()
    
    model.eval()
    fabric.print(f'======> The model is validating on Epoch {epoch}')
    
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    meters = [batch_time, data_time, losses, box_ap]
    meters_dict = {meter.name: meter for meter in meters}
    
    with torch.no_grad():
        end = time.time()
        for batch_idx, (prompt, image, bbox, gt_bbox, img_info) in enumerate(val_dataloader):
            predicted_bboxes = model(image, prompt)
            
            
    
def main(cfg):
    
    # set save dir
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_root, cfg.wandb.name, 'checkpoints')
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    cfg.log_dir = os.path.join(cfg.log_root, cfg.wandb.name, 'logs')
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    # set wandb
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, mode=cfg.wandb.mode, dir=cfg.log_dir,)
    
    csv_logger = CSVLogger(save_dir=cfg.log_dir, name=cfg.name)
    
    # init fabric
    fabric = Fabric(accelerator=cfg.accelerator, strategy=cfg.strategy, devices= cfg.devices, loggers=csv_logger)
    
    # set random seed
    fabric.seed_everything(cfg.seed)
    
    # load data
    train_dataset = PigCOCODataset(cfg, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=cfg.train.shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_dataset = PigCOCODataset(cfg, split='val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        shuffle=cfg.val.shuffle,
        num_workers=cfg.val.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # set up dataloaders for distributed training
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_loader, val_loader)
    
    # load model
    model = PigDetector(cfg)
    
    # set parameters for backpropagation
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # load optimizer
    optimizer = torch.optim.Adam(
        params=parameters,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    
    # set up optimizer for distributed training
    model, optimizer = fabric.setup(model, optimizer)
    
    if cfg.train.use_ema:
        model = EMA(model, cfg.train.ema_decay)

    # set up learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=cfg.train.T_0,
        T_mult=cfg.train.T_mult,
        eta_min=cfg.train.eta_min,
    )
    
    for epoch in range(cfg.train.epochs):
        train_one_epoch(fabric, model, optimizer, lr_scheduler, train_dataloader, cfg, epoch)
    
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pig Detection by image and text')
    parser.add_argument('--config', type=str, required=True, default='./config/refcoco.yaml')
    args = parser.parse_args()
    
    assert args.config is not None
    
    cfg = load_config(args.config)
    
    main(cfg)