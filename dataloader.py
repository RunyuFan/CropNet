import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from torchvision import transforms as T
import numpy as np
import cv2
def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='npy':
        return np.load(path)
    # elif type=='label':
    #     return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif type == 'label':
        return np.array(Image.open(path)).astype(int)


class GVGdataset(Dataset):
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        with open(txt,'r') as fh:
            file=[]
            for line in fh:
                line=line.strip('\n')

                labelpath = line.replace('img', 'label')
                imapath = line
                # print(imapath, labelpath)
                file.append((imapath, labelpath))


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self,index):

        img, label = self.file[index]
        # print(img, label)

        img = self.loader(img,type='img')
        label = self.loader(label,type='label')
        # print(label.tolist())

        if self.transform is not None:
            img=self.transform(img)

            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file)


if __name__ == "__main__":
    test_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_dataset=GVGdataset(txt='./data/train_patch.txt',transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True)
    for step,(img, label) in enumerate(test_loader):
        print(img.shape, label[:, :, 10, 10])
