import numpy as np
import torch
import torch.nn as nn
from utils import add_conv,DropBlock,FeatureAdaption,resblock,SPPLayer,upsample


def darknet_conv(in_ch, out_ch, ksize, stride=1, dilation_rate=1):
    
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (dilation_rate * (ksize - 1) + 1) // 2
    stage.add_module(
        'conv', 
        nn.Conv2d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=ksize, 
            stride=stride,
            padding=pad, 
            bias=False,
            dilation=dilation_rate
        )
    )
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class MultiScaleFusion(nn.Module):
    
    def __init__(self, v_planes=[256,512,1024], hiden_planes=512, scaled=True):
        super().__init__()
        self.up_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2) if scaled else nn.Sequential(),
                    darknet_conv(v_planes[-2] + hiden_planes // 2, hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
                ),
                nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2) if scaled else nn.Sequential(),
                    darknet_conv(v_planes[-1], hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes//2, hiden_planes // 2, 3)
                )
            ]
        )

        self.down_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                    darknet_conv(hiden_planes // 2 + v_planes[0], hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
                ),
                nn.Sequential(
                    nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                    darknet_conv(hiden_planes + v_planes[1], hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
                )
            ]
        )

        self.top_proj = darknet_conv(v_planes[-1] + hiden_planes // 2, hiden_planes, 1)
        self.mid_proj = darknet_conv(v_planes[1] + hiden_planes, hiden_planes, 1)
        self.bot_proj = darknet_conv(v_planes[0] + hiden_planes // 2, hiden_planes, 1)

    def forward(self, x):
        l,m,s = x
        # print(s)
        m = torch.cat([self.up_modules[1](s), m], 1)
        l = torch.cat([self.up_modules[0](m), l], 1)
        # out=self.out_proj(l)

        m = torch.cat([self.down_modules[0](l), m], 1)

        s = torch.cat([self.down_modules[1](m), s], 1)

        #top prpj and bot proj
        top_feat = self.top_proj(s)
        mid_feat = self.mid_proj(m)
        bot_feat = self.bot_proj(l)
        return [bot_feat,mid_feat,top_feat]
    
    
class YOLOv3Head(nn.Module):
    
    def __init__(
        self, 
        anch_mask, 
        n_classes, 
        stride, 
        in_ch=1024, 
        ignore_thre=0.7, 
        label_smooth = False, 
        rfb=False, 
        sep=False
    ):
        
        super(YOLOv3Head, self).__init__()
        self.anchors = [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (42, 119),
            (116, 90), (156, 198), (121, 240) 
        ]
        if sep:
            self.anchors = [
                (10, 13), (16, 30), (33, 23),
                (30, 61), (62, 45), (42, 119),
                (116, 90), (156, 198), (373, 326)
            ]

        self.anch_mask = anch_mask
        self.n_anchors = 4
        self.n_classes = n_classes
        self.guide_wh = nn.Conv2d(
            in_channels=in_ch,
            out_channels=2 * self.n_anchors, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.Feature_adaption=FeatureAdaption(in_ch, in_ch, self.n_anchors, rfb, sep)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.n_anchors * (self.n_classes+5), 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.stride = stride
        self._label_smooth = label_smooth
        self.all_anchors_grid = self.anchors
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]

    def forward(self, xin, labels=None):

        wh_pred = self.guide_wh(xin)  #Anchor guiding

        if xin.type() == 'torch.cuda.HalfTensor': #As DCN only support FP32 now, change the feature to float.
            wh_pred = wh_pred.float()
            if labels is not None:
                labels = labels.float()
            self.Feature_adaption = self.Feature_adaption.float()
            self.conv = self.conv.float()
            xin = xin.float()

        feature_adapted = self.Feature_adaption(xin, wh_pred)

        output = self.conv(feature_adapted)
        wh_pred = torch.exp(wh_pred)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        image_size = fsize * self.stride
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        wh_pred = wh_pred.view(batchsize, self.n_anchors, 2 , fsize, fsize)
        wh_pred = wh_pred.permute(0, 1, 3, 4, 2).contiguous()

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0,1,3,4,2).contiguous()

        x_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4])).to(xin.device)
        y_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4])).to(xin.device)

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(
            np.broadcast_to(
                np.reshape(
                    masked_anchors[:, 0], 
                    (1, self.n_anchors-1, 1, 1)
                ), 
                [batchsize, self.n_anchors-1, fsize, fsize]
            )
        ).to(xin.device)
        h_anchors = dtype(
            np.broadcast_to(
                np.reshape(
                    masked_anchors[:, 1], 
                    (1, self.n_anchors-1, 1, 1)
                ), 
                [batchsize, self.n_anchors-1, fsize, fsize]
            )
        ).to(xin.device)

        default_center = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).type(dtype).to(xin.device)

        pred_anchors = torch.cat((default_center, wh_pred), dim=-1).contiguous()

        anchors_based = pred_anchors[:, :self.n_anchors-1, :, :, :]
        anchors_free = pred_anchors[:, self.n_anchors-1, :, :, :]
        anchors_based[...,2] *= w_anchors
        anchors_based[...,3] *= h_anchors
        anchors_free[...,2] *= self.stride*4
        anchors_free[...,3] *= self.stride*4
        pred_anchors[...,:2] = pred_anchors[...,:2].detach()
        pred = output.clone()
        pred[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(pred[...,np.r_[:2, 4:n_ch]])
        pred[...,0] += x_shift
        pred[...,1] += y_shift
        pred[...,:2] *= self.stride
        pred[...,2] = torch.exp(pred[...,2])*(pred_anchors[...,2])
        pred[...,3] = torch.exp(pred[...,3])*(pred_anchors[...,3])
        pred_new = pred.view(batchsize, -1, pred.size()[2] * pred.size()[3] ,n_ch).permute(0, 2, 1, 3)
        refined_pred = pred.view(batchsize, -1, n_ch)
        return refined_pred.data,pred_new.data


class YOLOv3(nn.Module):

    def __init__(self, num_classes=80, ignore_thre=0.7, label_smooth=False, rfb=False):

        super(YOLOv3, self).__init__()
        self.module_list = create_yolov3_modules(num_classes, ignore_thre, label_smooth, rfb)

    def forward(self, x, targets=None, epoch=0):
        
        output = []
        feature_output = []
        boxes_output = []
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [19, 28, 37]:
                feature_output.append(x)
                x,box_output = module(x)
                boxes_output.append(box_output)
                output.append(x)
            else:
                x = module(x)
            # route layers
            if i in [6, 8, 17, 26]:
                route_layers.append(x)
            if i == 19:
                x = route_layers[2]
            if i == 28:  # yolo 2nd
                x = route_layers[3]
            if i == 21:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 30:
                x = torch.cat((x, route_layers[0]), 1)

        return torch.cat(output, 1),feature_output,boxes_output


def create_yolov3_modules(num_classes, ignore_thre, label_smooth, rfb):
    """
    Build yolov3 layer modules.
    """
    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))           #0
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))          #1
    mlist.append(resblock(ch=64))                                           #2
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))         #3
    mlist.append(resblock(ch=128, nblocks=2))                               #4
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))        #5
    mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here     #6
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))        #7
    mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here     #8
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))       #9
    mlist.append(resblock(ch=1024, nblocks=4))                              #10

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=1, shortcut=False))              #11
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))       #12
    #SPP Layer
    mlist.append(SPPLayer())                                                #13
    mlist.append(add_conv(in_ch=2048, out_ch=512, ksize=1, stride=1))       #14
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))       #15
    mlist.append(DropBlock(block_size=1, keep_prob=1.0))                    #16
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))       #17
    
    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))       #18
    mlist.append(
        YOLOv3Head(
            anch_mask=[6, 7, 8], 
            n_classes=num_classes, 
            stride=32, 
            in_ch=1024,
            ignore_thre=ignore_thre,
            label_smooth=label_smooth, 
            rfb=rfb
        )
    )           #19

    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))        #20
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #21
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))        #22
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))        #23
    mlist.append(DropBlock(block_size=1, keep_prob=1.0))                    #24
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))               #25
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))        #26
    
    # 2nd yolo branch
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))        #27
    mlist.append(
        YOLOv3Head(
            anch_mask=[3, 4, 5], 
            n_classes=num_classes, 
            stride=16, 
            in_ch=512,
            ignore_thre=ignore_thre, 
            label_smooth=label_smooth, 
            rfb=rfb
        )
    )         #28
    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))        #29
    mlist.append(upsample(scale_factor=2, mode='nearest'))                  #30
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))        #31
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))        #32
    mlist.append(DropBlock(block_size=1, keep_prob=1.0))                    #33
    mlist.append(resblock(ch=256, nblocks=1, shortcut=False))               #34
    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))        #35
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))        #36
    mlist.append(
        YOLOv3Head(
            anch_mask=[0, 1, 2], 
            n_classes=num_classes,
            stride=8, 
            in_ch=256,
            ignore_thre=ignore_thre, 
            label_smooth=label_smooth, 
            rfb=rfb
        )
    )         #37

    return mlist


backbone_dict={'yolov3':YOLOv3}


def visual_encoder(cfg):
    
    vis_encoder = backbone_dict[cfg.vis_encoder](num_classes=cfg.class_num)
    
    return vis_encoder

