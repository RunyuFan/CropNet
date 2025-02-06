# from dataloader import UISdataset_MM
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
# import pretrainedmodels
# from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate
from model_fuse import LightSDNet
from torch.utils.data import DataLoader, Dataset
import time
from dataloader import GVGdataset
# from make_dataset import ImgDataset, readfile, readfile_add, ImgDataset_train
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
# import pytorch_ssim # pytorch_ssim.SSIM()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Boundary-Aware Feature Propagation
# from rmi import RMILoss
import sklearn.utils.class_weight as class_weight
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        if preds.shape[-2:] != labels.shape[-2:]:
            preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=False)

        n_min = 100000  # labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, list):
            return sum([self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)



def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_txt = "./data/train_patch.txt"
    val_txt = "./data/val_patch.txt"
    # test_txt = "./data/testSSOUISdataset_all.txt"

    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()
                                                ])
    val_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # test_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset=GVGdataset(txt=train_txt,transform=train_transform)
    val_dataset=GVGdataset(txt=val_txt,transform=val_transform)
    # test_dataset=UISdataset_MM(txt=test_txt,transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=1,pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8,pin_memory=True)

    print("Train numbers:{:d}".format(len(train_dataset)))
    print("val numbers:{:d}".format(len(val_dataset)))

    model2 = torch.load('.\model\GVG-2.pth')  # MTSDformer(2)

    print('model2 parameters:', sum(p.numel() for p in model2.parameters() if p.requires_grad))

    input1 = torch.randn(16, 3, 256, 256).float().cuda()
    flops, params = profile(model2, inputs=(input1, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

#     # model1 = model1.to(device)
#     model2 = model2.to(device)
#     # model3 = model3.to(device)
#     # cost1 = OhemCrossEntropy().to(device)
#     # class_weights = torch.tensor([0.6601, 2.0615], dtype=torch.float).to(device)
#     # cost1 = nn.CrossEntropyLoss(weight=class_weights).to(device)  # [0.6601, 2.0615]
#     cost1 = nn.CrossEntropyLoss().to(device)  # [0.6601, 2.0615]
#     # cost2 = nn.BCEWithLogitsLoss().to(device)
#
#     # cost1 = RMILoss(with_logits=False).to(device)
#     # cost2 = RMILoss(with_logits=False).to(device)
#     # loss = RMILoss(with_logits=True)
#     # ssim_loss = pytorch_ssim.SSIM()
#             # out = self.backbone(h_rs)
#             # out = self.decode_head(out)
#     optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-6)
#     # optimizer2 = torch.optim.Adam([{'params': model2.backbone.parameters(), 'lr': 1e-4}, {'params': model2.decode_head.parameters()}], lr=args.lr)
#
#     # model2.eval()
#     # model3.eval()
#
#     # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
#
#     print('------------------------------------')
#     print('start to evaluate in the testing set')
#     print('------------------------------------')
#
#     model2.eval()
#     # model3.eval()
#
#     # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
#     classes = ['_background_', 'bareland', 'wasteland', 'soybean', 'rice', 'rape', 'corn', 'wheat', 'fallow'] # ('住宅区', '公共服务区域', '商业区', '城市绿地', '工业区')
#
#     hist = torch.zeros(args.num_class, args.num_class).to(device)
#
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(device)
#
#             # mmdata = data[1].to(device)
#             # print(images.shape)
#             labels = labels.to(device, dtype=torch.int64).squeeze(1)
#             # instance_label = instance_label.to(device, dtype=torch.int64)
#             # mmdata = mmdata.clone().detach().float()
#             images = images.clone().detach().float()
#             # labels = labels.clone().detach().Long()
#
#             # Forward pass
#             # outputs1 = model1(images)
#             out_0, out, outputs2 = model2(images)
#             # print(labels.shape)
#
#             # Forward pass
#             # outputs1 = model1(images)
#             # outputs2 = model2(images, mmdata)
#             # outputs3 = model3(images)
#             # print(outputs2.shape)
#             # loss1 = cost1(outputs1, labels)
#             preds = outputs2.softmax(dim=1).argmax(dim=1)
#             # print(preds.shape)
#
#             keep = labels != 1000
#             hist += torch.bincount(labels[keep] * args.num_class + preds[keep], minlength=args.num_class**2).view(args.num_class, args.num_class)
#
#     ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
#     miou = ious[~ious.isnan()].mean().item()
#     ious = ious.cpu().numpy().tolist()
#     miou = miou * 100
#
#     Acc = hist.diag() / hist.sum(1)
#     mOA = hist.diag().sum() / hist.sum() * 100
#
#     table = {
#         'Class': classes,
#         'IoU': ious,
#         'Acc': Acc,
#         # 'mOA': mOA
#     }
#
#     print(tabulate(table, headers='keys'))
#     print(f"\nOverall mIoU: {miou:.2f}")
#     print(f"\nOverall mOA: {mOA:.2f}")
#
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=9, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    args = parser.parse_args()

    main(args)
