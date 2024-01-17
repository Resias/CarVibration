
import os
import torch
import torchvision
from torch.utils.data import  DataLoader
from dataset import UntrimmedDataset
from model import CPD_SSL
data_root = os.path.join(os.getcwd(), 'slice_data')
n_fft = 16
hop_length = int(n_fft/2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device ='cpu'
feature_size = 32



dataset = UntrimmedDataset(root_dir=data_root,
                               kernel_size= 64,
                               stride=32,
                               device=device,
                               n_fft=n_fft,
                               hop_length=hop_length)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

cpd = CPD_SSL(backbone='RegNet', feature_size=32, device=device)
cpd.backbone.to(device)

from torchvision import transforms, utils
transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),  # 이미지 크기를 256x256으로 조정
])

cpd.train(dataloader, epoch=200, transforms=transform)