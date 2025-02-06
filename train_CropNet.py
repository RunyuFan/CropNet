# from dataloader import UISdataset_MM
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121

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
    val_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # test_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = GVGdataset(txt=train_txt, transform=train_transform)
    val_dataset = GVGdataset(txt=val_txt, transform=val_transform)
    # test_dataset=UISdataset_MM(txt=test_txt,transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=1,pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8,pin_memory=True)

    print("Train numbers:{:d}".format(len(train_dataset)))
    print("val numbers:{:d}".format(len(val_dataset)))

    model2 = LightSDNet(9)

    print('model2 parameters:', sum(p.numel() for p in model2.parameters()))
    Trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f'Trainable params: {Trainable_params/ 1e6}M')

    # model1 = model1.to(device)
    model2 = model2.to(device)

    cost1 = nn.CrossEntropyLoss().to(device)  # [0.6601, 2.0615]

    optimizer2 = torch.optim.AdamW([{'params': model2.backbone.parameters(), 'lr': 1e-4}, {'params': model2.decode_path.parameters()},
    {'params': model2.SpatialPath.parameters()}, {'params': model2.FeatureFusionModule.parameters()}, {'params': model2.conv_fc.parameters()}, {'params': model2.conv_sp_fc.parameters()}], lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)



    # best_acc_1 = 0.
    miou_max = 0.
    best_epoch = 0
    # best_acc_3 = 0.
    # alpha = 1
    for epoch in range(1, args.epochs + 1):
        # model1.train()
        model2.train()
        # model3.train()
        # start time
        start = time.time()
        index = 0
        for images, labels in train_loader:
            images = images.to(device)

            # mmdata = data[1].to(device)
            # print(images.shape)
            labels = labels.to(device, dtype=torch.int64).squeeze(1)
            # edges = data[2].to(device, dtype=torch.int64)
            # print(images.shape, labels.shape)
            # instance_label = instance_label.to(device, dtype=torch.int64)
            # mmdata = mmdata.clone().detach().float()
            #print(images.shape)
            images = images.clone().detach().float()
            #print(images.shape)
            # m = nn.Sigmoid()
            # labels = labels.clone().detach().Long()

            # Forward pass
            # outputs1 = model1(images)
            out_0, out, outputs2 = model2(images)
            # print(out_0.shape, out.shape, outputs2.shape)
            # outputs3 = model3(images)
            # loss1 = cost1(outputs1, labels)
            # print(outputs2.shape, labels.squeeze(1).shape, instance_out.shape, instance_label.squeeze(1).shape)
            loss = 0.1*cost1(out_0, labels) + 0.1*cost1(out, labels) + 0.8*cost1(outputs2, labels)
            # loss2 = 0.02*cost2(out_edge, edges.unsqueeze(1).float()) # torch.nn.functional.one_hot(edges)
            # loss = loss1 + loss2
            # loss3 = cost3(outputs3, labels)

            # if index % 10 == 0:
                # print (loss)
            # Backward and optimize
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer3.zero_grad()
            # loss1.backward()
            loss.backward()
            # loss3.backward()
            # optimizer1.step()
            optimizer2.step()
            # optimizer3.step()
            index += 1

        # scheduler_poly_lr_decay.step()
        if epoch % 1 == 0:
            end = time.time()
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss1.item(), (end-start) * 2))
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss.item(), (end-start) * 2))
            # print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss3.item(), (end-start) * 2))

            # model1.eval()
            model2.eval()
            # model3.eval()

            # classes = ('bareland', 'cropland', 'forest', 'impervious', 'shrub', 'water')
            classes = ['_background_', 'bareland', 'wasteland', 'soybean', 'rice', 'rape', 'corn', 'wheat', 'fallow'] # ('住宅区', '公共服务区域', '商业区', '城市绿地', '工业区')

            hist = torch.zeros(args.num_class, args.num_class).to(device)

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)

                    # mmdata = data[1].to(device)
                    # print(images.shape)
                    labels = labels.to(device, dtype=torch.int64).squeeze(1)
                    # instance_label = instance_label.to(device, dtype=torch.int64)
                    # mmdata = mmdata.clone().detach().float()
                    images = images.clone().detach().float()
                    # labels = labels.clone().detach().Long()

                    # Forward pass
                    # outputs1 = model1(images)
                    out_0, out, outputs2 = model2(images)
                    # print(labels.shape)

                    # Forward pass
                    # outputs1 = model1(images)
                    # outputs2 = model2(images, mmdata)
                    # outputs3 = model3(images)
                    # print(outputs2.shape)
                    # loss1 = cost1(outputs1, labels)
                    preds = outputs2.softmax(dim=1).argmax(dim=1)
                    # print(preds.shape)

                    keep = labels != 1000
                    hist += torch.bincount(labels[keep] * args.num_class + preds[keep], minlength=args.num_class**2).view(args.num_class, args.num_class)

            ious = hist.diag() / (hist.sum(0) + hist.sum(1) - hist.diag())
            miou = ious[~ious.isnan()].mean().item()
            ious = ious.cpu().numpy().tolist()
            miou = miou * 100

            Acc = hist.diag() / hist.sum(1)
            mOA = hist.diag().sum() / hist.sum() * 100

            table = {
                'Class': classes,
                'IoU': ious,
                'Acc': Acc,
                # 'mOA': mOA
            }

            print(tabulate(table, headers='keys'))
            print(f"\nOverall mIoU: {miou:.2f}")
            print(f"\nOverall mOA: {mOA:.2f}")

        if  miou > miou_max:
            print('save new best miou', miou)
            torch.save(model2, os.path.join(args.model_path, 'GVG-1.pth'))  # GVG-2.pth
            miou_max = miou
            best_epoch = epoch

        print('Current best iou', miou_max, best_epoch)
        print("-----------------------------------------")
    # print('save new best acc_3', best_acc_3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=9, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    # parser.add_argument("--net", default='ResNet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    args = parser.parse_args()

    main(args)
