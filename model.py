import torch
import torchvision
import torch.optim as optim
import os
import json
import pandas as pd

class CPD_SSL():
    def __init__(self, backbone, feature_size, device):
        self.backbone = self.backbone_load(backbone, feature_size, device)
        self.device = device
        
    def backbone_load(self, backbone, feature_size, device):
        
        # 1. SqueezeNet
        if backbone == 'SqeezeNet':
            squeeze_net = torchvision.models.squeezenet1_1(progress=True).to(device)
            squeeze_net.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
                torch.nn.Flatten(),
                torch.nn.Linear(512, feature_size, bias=True))
            return squeeze_net
            
        # 2. ShuffleNet
        elif backbone == 'ShuffleNet':
            shuffle_net = torchvision.models.shufflenet_v2_x2_0().to(device)
            shuffle_net.fc = torch.nn.Linear(in_features=2048, out_features=feature_size, bias=True)
            return shuffle_net
            
        # 3. RegNet
        elif backbone == 'RegNet':
            reg_net = torchvision.models.regnet_y_1_6gf().to(device)
            reg_net.fc = torch.nn.Linear(in_features=888, out_features=feature_size, bias=True)
            return reg_net
            
        # 4. MobileNet
        elif backbone == 'MobileNet':
            mobile_net = torchvision.models.mobilenet_v3_large().to(device)
            mobile_net.classifier = torch.nn.Sequential(
                torch.nn.Linear(960, 1280, bias=True),
                torch.nn.Linear(1250, feature_size, bias=True))
            return mobile_net

        # 5. EfficientNet
        elif backbone == 'EfficientNet':
            efficient_net = torchvision.models.efficientnet_b2().to(device)
            efficient_net.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.3, inplace=True),
                torch.nn.Linear(in_features=1408, out_features=feature_size, bias=True))
            return efficient_net

        # 6. MnasNet
        elif backbone == 'MnasNet':
            mnas_net = torchvision.models.mnasnet1_3().to(device)
            mnas_net.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=1280, out_features=feature_size, bias=True))
            return mnas_net
        else:
            print(f'Error : Unspportable Backbone - {backbone}')

    def train(self, train_loader, epoch, transforms):
        optimier = optim.Adam(self.backbone.parameters(), lr=0.001)
        
        self.experiment_name = self.backbone.__class__.__name__
        print(f'backbone : {self.backbone.__class__.__name__}')
        self.output_path = os.path.join(os.getcwd(), 'outputs_nfft16_h2_b64_k64_s32' ,self.experiment_name)
        if os.path.isdir(self.output_path):
            print(f'Error : path{self.output_path} is already exist')
            exit()
        os.makedirs(self.output_path)
        
        result ={'Epoch' : [],
                'loss_epoch' : [],
                'mean_pos' : [],
                'mean_neg' : [],
        }
        
        result_df = pd.DataFrame(result)
        best_loss = 1000000000000000
        for i in range(epoch):
            loss_epoch, mean_pos_epoch, mean_neg_epoch = self.train_one_epoch(data_loader=train_loader, optimizer=optimier, transforms=transforms)
            
            print(f'Epoch : {i}/{epoch} | loss_epoch : {loss_epoch} | mean_pos : {mean_pos_epoch} | mean_neg : {mean_neg_epoch}')
            
            new_data = {
                'Epoch': [int(i)],
                'loss_epoch': [loss_epoch],
                'mean_pos': [mean_pos_epoch],
                'mean_neg' : [mean_neg_epoch]
                }
            new_data = pd.DataFrame(new_data)
            result_df = pd.concat([result_df, new_data])
            
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            
            if i%10 == 0:
                torch.save(self.backbone.state_dict(), os.path.join(self.output_path, f'Epoch_{i}.pth'))
            if loss_epoch < best_loss:
                print(f'Best Loss : {loss_epoch}')
                torch.save(self.backbone.state_dict(), os.path.join(self.output_path, f'Best_Loss.pth'))
                best_loss = loss_epoch
                
        self.save_dataframe_as_json(result_df)
        
    def save_dataframe_as_json(self, dataframe):
        path = str(self.output_path) + '/result.json'
        dataframe.to_json(path, orient='records', indent=4)
            
            
    def train_one_epoch(self, data_loader, optimizer, transforms=None):
        
        loss_epoch = 0.0
        mean_pos_epoch = 0.0
        mean_neg_epoch = 0.0
        
        for idx, batch in enumerate(data_loader):
            
            data_stft, class_tensor, is_new = batch
            
            data_stft.to(self.device)
            if transforms is not None:
                data_stft = transforms(data_stft)
            
            loss_step, mean_pos, mean_neg = self.train_one_step(data_stft, optimizer)
            
            loss_epoch += loss_step
            mean_pos_epoch += mean_pos
            mean_neg_epoch += mean_neg
        
        loss_epoch /= len(data_loader)
        mean_pos_epoch /= len(data_loader)
        mean_neg_epoch /= len(data_loader)
        
        return loss_epoch, mean_pos_epoch, mean_neg_epoch
        
        
    def train_one_step(self, batch, optimizer):
        
        loss = InfoNCE(negative_mode='paired').to(self.device)
        batch = batch.to(self.device)
        
        batch = self.backbone(batch).to(self.device)
        
        query = batch[:-1]
        positive_pair = batch[1:]
        negative_pair = []

        for i in range(len(batch)-1):
            neg_i = []
            for j in range(len(batch)):
                if i!=j and j!=i+1:
                    neg_i.append(batch[j])
            neg_i = torch.stack(neg_i)
            negative_pair.append(neg_i)

        negative_pair = torch.stack(negative_pair).to(self.device)


        loss_step, mean_pos, mean_neg = loss(query, positive_pair, negative_pair)

        optimizer.zero_grad()
        loss_step.backward()
        optimizer.step()
        

        
        return loss_step.item(), mean_pos.item(), mean_neg.item()
    
    def valid_one_epoch(self, data_loader, threshold = 0.0, transforms=None):
        
        best_acc = 0.0
        self.backbone.eval()
        
        true_correct = 0
        false_correct = 0
        total = 0
        
        setting_threshold = threshold

        anomally = 0.0
        with torch.no_grad():
            for index, batches in enumerate(data_loader):
                
                data_stft, labels, is_new = batches
                
                for label in labels:
                    start_l = label[0]
                    for l in label:
                        if start_l == l: continue
                        else:
                            anomally +=1
                            break
                
                data_stft.to(self.device)
                if transforms is not None:
                    data_stft = transforms(data_stft)
                
                batch = self.backbone(data_stft)
                for idx in range(len(batch)):                               # 각 배치별
                    
                    if idx + 1 >= len(batch):
                        break
                    
                    cos_sim_f = nn.CosineSimilarity(dim=0)
                    cos_sim = cos_sim_f(batch[idx],batch[idx+1])            # 근사한 2쌍 cosine similarity
                    
                    if cos_sim < setting_threshold:                         # 해당 배치가 threshold 이하인지
                        l1_0 = labels[idx][0]
                        # l0 = len(set(labels[idx].unique().numpy()))
                        tf_flag1 = True 
                        
                        for l_ in labels[idx]:                               # 실제 CP인지 확인
                            if l1_0 == l_: continue
                            else:
                                tf_flag1 = False
                                break
                        
                        
                        l2_0 = labels[idx + 1][0]
                        # l0 = len(set(labels[idx].unique().numpy()))
                        tf_flag2 = True
                        
                        for l__ in labels[idx + 1]:                               # 실제 CP인지 확인
                            if l2_0 == l__: continue
                            else:
                                tf_flag2 = False
                                break
                        
                        if tf_flag1 == False or tf_flag2 == False:
                            true_correct += 1
                        else:
                            false_correct += 1
                
                # correct += (predicted == targets).sum().item()
                print(labels)
                print(true_correct, anomally)
                print(f'[Test] index: {index + 1} | true Acc: {true_correct / anomally * 100:.4f}')
            
            print(f'[Test] epoch: {1} | true Acc: {true_correct / anomally * 100:.4f}')



import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        try:
            logits = torch.cat([positive_logit, negative_logits], dim=1)
        except:
            print(negative_logits)
            exit()
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction), torch.mean(positive_logit), torch.mean(negative_logits)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]