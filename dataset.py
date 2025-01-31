import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import librosa
import numpy as np
from interpolate import interpolate_data

class UntrimmedDataset(Dataset):
    def __init__(self, root_dir, kernel_size, stride, device, n_fft, hop_length):
        self.root_dir = root_dir
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.data = pd.DataFrame(columns=['x', 'y', 'z', 'uptime', 'class'])
        
        files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.csv')]
        
        for file in files:
            data_pd = pd.read_csv(file)
            data_length = len(data_pd)
            stride_n = (data_length - kernel_size) // stride
            adjusted_length = kernel_size + (stride_n * stride)
            data_pd = data_pd.head(adjusted_length)
            
            fileinfo_pd = pd.DataFrame({'end_file': [False] * adjusted_length})
            fileinfo_pd.iloc[0] = True

            data_pd = data_pd[['x', 'y', 'z', 'uptime', 'class']]
            data_pd = pd.concat([data_pd, fileinfo_pd], axis=1)
            
            x_tensor = torch.tensor(data_pd['x'].values, dtype=torch.float32)
            y_tensor = torch.tensor(data_pd['y'].values, dtype=torch.float32)
            z_tensor = torch.tensor(data_pd['z'].values, dtype=torch.float32)
            class_tensor = torch.tensor(data_pd['class'].values, dtype=torch.float32)

            uptime_tensor = torch.tensor(data_pd['uptime'].values, dtype=torch.float32)
            data_tensor = torch.tensor(data_pd[['x', 'y', 'z','class']].values, dtype=torch.float32)
            # print(f'before interpolation : {x_tensor.shape}')
            interpolated_x = self.interpolate(uptime_tensor.numpy(),x_tensor.numpy()) 
            interpolated_y = self.interpolate(uptime_tensor.numpy(),y_tensor.numpy()) 
            interpolated_z = self.interpolate(uptime_tensor.numpy(),z_tensor.numpy()) 
            interpolated_class = self.interpolate(uptime_tensor.numpy(),class_tensor.numpy())
            # print(f'after_interpolation : {interpolated_x.shape}')
            interpolated_data = pd.DataFrame({
                'x': interpolated_x,
                'y': interpolated_y,
                'z': interpolated_z,
                'class' : interpolated_class,
                'uptime': uptime_tensor.numpy()  # 새로운 'uptime' 열 추가
                })
            
            current_data = pd.concat([pd.DataFrame(interpolated_data, columns=['x', 'y', 'z','class']), data_pd[['uptime', 'end_file']]], axis=1)
            self.data = pd.concat([self.data, current_data])

        self.data_len = int((len(self.data) - kernel_size) / stride + 1)
        
    def interpolate(self,uptime,data):
        interpolated_data = interpolate_data(uptime, data, new_interval=1e-4 )
        return interpolated_data
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.kernel_size
        x_data = torch.tensor(self.data[['x']].values[start_idx:end_idx]).view(-1)
        y_data = torch.tensor(self.data[['y']].values[start_idx:end_idx]).view(-1)
        z_data = torch.tensor(self.data[['z']].values[start_idx:end_idx]).view(-1)

        x_stft = torch.stft(input=x_data,n_fft=self.n_fft,hop_length=self.hop_length,return_complex=True).to(self.device)
        y_stft = torch.stft(input=y_data,n_fft=self.n_fft,hop_length=self.hop_length,return_complex=True).to(self.device)
        z_stft = torch.stft(input=z_data,n_fft=self.n_fft,hop_length=self.hop_length,return_complex=True).to(self.device)

        data_stft = torch.stack((x_stft,y_stft,z_stft)).float().to(self.device)
        other_columns = self.data[['uptime', 'class', 'end_file']]

        #class_tensor = torch.tensor(self.data['class'][start_idx:end_idx])
        class_tensor = torch.tensor([int(value) for value in self.data['class'][start_idx:end_idx]]).to(self.device)

        is_new = False
        if self.data['end_file'].isin([True]).any():
            is_new = True

        #return torch.tensor(data_stft, dtype=torch.float32), class_tensor, is_new
        return data_stft, class_tensor, is_new

class TrimmedDataset(Dataset):
    def __init__(self, root_dir, kernel_size, stride, device, n_fft, hop_length, isTrain=True):
        self.root_dir = root_dir
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        data_root = os.path.join(os.getcwd(), root_dir)

        train_list = []
        test_list = []

        self.data = pd.DataFrame(columns=['x', 'y', 'z', 'uptime', 'class'])
        for idx, class_dir in enumerate(os.listdir(data_root)):

            class_dir = os.path.join(data_root, class_dir)
            
            for csv_name in ['1.csv', '2.csv']:
                train_csv = os.path.join(class_dir, csv_name)
            
            for csv_name in ['3.csv', '4.csv']:
                test_csv = os.path.join(class_dir, csv_name)
            
            train_list.append(train_csv)
            test_list.append(test_csv)
        
        if isTrain:
            data_list = train_list
        else:
            data_list = test_list
        
        for idx, file in enumerate(data_list):
            data_pd = pd.read_csv(file)
            data_length = len(data_pd)
            stride_n = (data_length - kernel_size) // stride
            adjusted_length = kernel_size + (stride_n * stride)
            data_pd = data_pd.head(adjusted_length)
            
            fileinfo_pd = pd.DataFrame({'end_file': [False] * adjusted_length})
            fileinfo_pd.iloc[0] = True

            data_pd = data_pd[['x', 'y', 'z', 'uptime']]
            data_pd = pd.concat([data_pd, fileinfo_pd], axis=1)
            
            x_tensor = torch.tensor(data_pd['x'].values, dtype=torch.float32)
            y_tensor = torch.tensor(data_pd['y'].values, dtype=torch.float32)
            z_tensor = torch.tensor(data_pd['z'].values, dtype=torch.float32)
            # class_tensor = torch.tensor(data_pd['class'].values, dtype=torch.float32)

            uptime_tensor = torch.tensor(data_pd['uptime'].values, dtype=torch.float32)
            # print(f'before interpolation : {x_tensor.shape}')
            interpolated_x = self.interpolate(uptime_tensor.numpy(),x_tensor.numpy()) 
            interpolated_y = self.interpolate(uptime_tensor.numpy(),y_tensor.numpy()) 
            interpolated_z = self.interpolate(uptime_tensor.numpy(),z_tensor.numpy()) 
            interpolated_class = [idx for _ in range(len(interpolated_x))]
            
            # print(f'after_interpolation : {interpolated_x.shape}')
            interpolated_data = pd.DataFrame({
                'x': interpolated_x,
                'y': interpolated_y,
                'z': interpolated_z,
                'class' : interpolated_class,
                'uptime': uptime_tensor.numpy()  # 새로운 'uptime' 열 추가
                })
            
            current_data = pd.concat([pd.DataFrame(interpolated_data, columns=['x', 'y', 'z','class']), data_pd[['uptime', 'end_file']]], axis=1)
            self.data = pd.concat([self.data, current_data])

        self.data_len = int((len(self.data) - kernel_size) / stride + 1)
        
    def interpolate(self,uptime,data):
        interpolated_data = interpolate_data(uptime, data, new_interval=1e-4 )
        return interpolated_data
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.kernel_size
        x_data = torch.tensor(self.data[['x']].values[start_idx:end_idx]).view(-1)
        y_data = torch.tensor(self.data[['y']].values[start_idx:end_idx]).view(-1)
        z_data = torch.tensor(self.data[['z']].values[start_idx:end_idx]).view(-1)

        x_stft = torch.stft(input=x_data,n_fft=self.n_fft,hop_length=self.hop_length,return_complex=True)
        y_stft = torch.stft(input=y_data,n_fft=self.n_fft,hop_length=self.hop_length,return_complex=True)
        z_stft = torch.stft(input=z_data,n_fft=self.n_fft,hop_length=self.hop_length,return_complex=True)

        data_stft = torch.stack((x_stft,y_stft,z_stft)).float()

        #class_tensor = torch.tensor(self.data['class'][start_idx:end_idx])
        class_tensor = torch.tensor([int(value) for value in self.data['class'][start_idx:end_idx]])

        is_new = False
        if self.data['end_file'].isin([True]).any():
            is_new = True

        #return torch.tensor(data_stft, dtype=torch.float32), class_tensor, is_new
        return data_stft, class_tensor, is_new


if __name__ == '__main__':
    
    data_root = os.path.join(os.getcwd(), 'Untrimmed_accline')
    n_fft = 32  # STFT의 FFT 크기 설정
    hop_length = int(n_fft/4)  # STFT의 hop 길이 설정
    
    # dataset = UntrimmedDataset(root_dir=data_root,
    #                            kernel_size=512,
    #                            stride=256,
    #                            device='cpu',
    #                            n_fft=n_fft,
    #                            hop_length=hop_length)
    
    # dataloader = DataLoader(dataset, batch_size=4)
    

    # for i, (data, class_tensor, is_new) in enumerate(dataloader):
    #     print(f"Batch {i + 1}:")
    #     print("Data shape:", data.shape)
    #     print("Class tensor:", class_tensor)
    #     print("Is new:", is_new)
        
    #     if i == 1:  # 처음 3개의 배치만 확인
    #         break
    
    data_root = os.path.join(os.getcwd(), 'Trimmed_accline')
    dataset = TrimmedDataset(root_dir=data_root,
                               kernel_size=64,
                               stride=32,
                               device='cpu',
                               n_fft=n_fft,
                               hop_length=hop_length)
    
    dataloader = DataLoader(dataset, batch_size=1)
    
    print(f'len(dataloader) : {len(dataloader)}')

    for i, (data, class_tensor, is_new) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print("Data shape:", data.shape)
        print("Class tensor:", class_tensor)
        print("Is new:", is_new)
        
        if i == 1:  # 처음 3개의 배치만 확인
            break