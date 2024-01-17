
import os
import torch
import torchvision
from torch.utils.data import  DataLoader
from dataset import UntrimmedDataset
from model_val import CPD_SSL


data_root = os.path.join(os.getcwd(), 'slice_data')
model_name='MnasNet'
n_fft = 32
hop_length = int(n_fft/2)
#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_size = 32
batch_size = 64

dataset = UntrimmedDataset(root_dir=data_root,
                            kernel_size= 64,
                            stride=32,
                            device=device,
                            n_fft=n_fft,
                            hop_length=hop_length)

dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

pth_path = r'/home/CarVibration-main/2024_01_09_result/outputs_nfft32_h_16_b_64_k_64_s_32/MNASNet/Best_Loss.pth'
file_name = pth_path.split("\\")[-1]
cpd = CPD_SSL(backbone=model_name, feature_size=feature_size, device=device)
cpd.load_model(pth_path=pth_path)
cpd.backbone.to(device)

from torchvision import transforms, utils
transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),  # 이미지 크기를 256x256으로 조정
])


cpd.valid_one_epoch(dataloader,file_name,threshold=0.6, transforms=transform)