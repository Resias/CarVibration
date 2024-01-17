import os
import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import  DataLoader
from dataset import UntrimmedDataset, TrimmedDataset
from model_val import CPD_SSL


data_root = os.path.join(os.getcwd(), 'Trimmed_accline')
n_fft = 32
hop_length = int(n_fft/4)
device = 'cpu'
feature_size = 32
batch_size = 1

        # nfft, hop, batch, kernel, stride
config = [[32, 8, 32, 32, 16],
          [32, 8, 32, 64, 32],
          [32, 8, 64, 32, 16],
          [32, 8, 64, 64, 32],
          [32, 8, 128, 32, 16],
          [32, 8, 128, 64, 32],
          [32, 16, 32, 32, 16],
          [32, 16, 32, 64, 32],
          [32, 16, 64, 32, 16],
          [32, 16, 64, 64, 32],
          [32, 16, 128, 32, 16],
          [32, 16, 128, 64, 32],
#error     [64, 32, 32, 32, 16],
          [64, 16, 32, 64, 32],
#error     [64, 16, 64, 32, 16],
          [64, 32, 32, 64, 32],
#error     [64, 32, 64, 32, 16],
#error     [64, 32, 64, 64, 32],
#error     [64, 32, 128, 32, 16],
          [64, 16, 128, 64, 32],
          [64, 32, 128, 64, 32]
          ]

def test_classification(config):

    best_setting = {'SqueezeNet' : {'nfft' : [0],'hop length' : [0], 'batch size' : [0], 'kernel size' : [], 'stride size' : [],
                                    'acc' : [0], 'precision' : [0], 'recall' : [0]},
                        'ShuffleNetV2' : {'nfft' : [0],'hop length' : [0], 'batch size' : [0], 'kernel size' : [], 'stride size' : [],
                                    'acc' : [0], 'precision' : [0], 'recall' : [0]},
                        'RegNet' : {'nfft' : [0],'hop length' : [0], 'batch size' : [0], 'kernel size' : [], 'stride size' : [],
                                    'acc' : [0], 'precision' : [0], 'recall' : [0]},
                        'MobileNetV3' : {'nfft' : [0],'hop length' : [0], 'batch size' : [0], 'kernel size' : [], 'stride size' : [],
                                    'acc' : [0], 'precision' : [0], 'recall' : [0]},
                        'EfficientNet' : {'nfft' : [0],'hop length' : [0], 'batch size' : [0], 'kernel size' : [], 'stride size' : [],
                                    'acc' : [0], 'precision' : [0], 'recall' : [0]},
                        'MNASNet' : {'nfft' : [0],'hop length' : [0], 'batch size' : [0], 'kernel size' : [], 'stride size' : [],
                                    'acc' : [0], 'precision' : [0], 'recall' : [0]},
                        }
    
    for c in config:
        c_n_fft = c[0]
        c_hop_length = c[1]
        c_batch_size = c[2]
        c_kernel_size = c[3]
        c_stride_size = c[4]
        
        backbones = ['SqeezeNet','ShuffleNet','RegNet','MobileNet','EfficientNet','MnasNet']
        folder_name = '2024_01_11_result/outputs_nfft' + str(c_n_fft) + '_h_' + str(c_hop_length) + '_b_' + str(c_batch_size) + '_k_' + str(c_kernel_size) + '_s_' + str(c_stride_size)

        
        for backbone in backbones:
            
            train_dataset = TrimmedDataset(root_dir=data_root,
                                        kernel_size= 64,
                                        stride=16,
                                        device=device,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        isTrain=False)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)


            test_dataset = TrimmedDataset(root_dir=data_root,
                                        kernel_size= 64,
                                        stride=16,
                                        device=device,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        isTrain=True)

            test_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)


            cpd = CPD_SSL(backbone=backbone, feature_size=feature_size, device=device)
            try:
                cpd.load_model_auto(folder_name)
            except:
                print("CPD load failed")
                continue
            transform = transforms.Compose([
                transforms.Resize((128, 128), antialias=True),
                ])

            cpd.train_classifier(data_loader=train_dataloader, transforms=transform)

            accuracy, precision, recall = cpd.test_classifier(data_loader=test_dataloader, transforms=transform)
            
            dir = os.path.join(os.getcwd(), folder_name, cpd.backbone.__class__.__name__)
            with open(f'result_{backbone}.txt','a') as f:
                f.write(f'CPD_config:\n')
                f.write(f' nfft : {c_n_fft}')
                f.write(f'\n hop_length : {c_hop_length}')
                f.write(f'\n batch_size : {c_batch_size}')
                f.write(f'\n kernel_size : {c_kernel_size}')
                f.write(f'\n stride_size : {c_stride_size}')
                f.write(f'\n\naccuracy : {accuracy} ')
                f.write(f'recall : {recall} ')
                f.write(f'precision : {precision}\n\n')
            
            torch.save(cpd, os.path.join(os.getcwd(),folder_name,cpd.backbone.__class__.__name__, f'Classfication.pth'))
            
            if best_setting[cpd.backbone.__class__.__name__]['acc'] < accuracy:
                best_setting[cpd.backbone.__class__.__name__]['acc'] = accuracy
                best_setting[cpd.backbone.__class__.__name__]['recall'] = recall
                best_setting[cpd.backbone.__class__.__name__]['precision'] = precision
                
                best_setting[cpd.backbone.__class__.__name__]['nfft'] = c_n_fft
                best_setting[cpd.backbone.__class__.__name__]['hop length'] = c_hop_length
                best_setting[cpd.backbone.__class__.__name__]['kernel size'] = c_kernel_size
                best_setting[cpd.backbone.__class__.__name__]['stride size'] = c_stride_size
                best_setting[cpd.backbone.__class__.__name__]['batch size'] = c_batch_size
                
    models = ['SqueezeNet','ShuffleNetV2','RegNet','MobileNetV3','EfficientNet','MNASNet']
        
    with open(f'best_result.txt','w') as f:
        for m in models:
            f.write(f'Best {m} CPD config:\n')
            f.write(f' nfft : {best_setting[m]["nfft"]}')
            f.write(f'\n hop_length : {best_setting[m]["hop length"]}')
            f.write(f'\n batch_size : {best_setting[m]["batch size"]}')
            f.write(f'\n kernel_size : {best_setting[m]["kernel size"]}')
            f.write(f'\n stride_size : {best_setting[m]["stride size"]}')
            f.write(f'\n\naccuracy : {best_setting[m]["acc"]} ')
            f.write(f'recall : {best_setting[m]["recall"]} ')
            f.write(f'precision : {best_setting[m]["precision"]}\n\n')
            
                
                

test_classification(config)
