# -*- coding: utf-8 -*-
"""
Pytorch-related functions and classes for classification
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from tqdm import tqdm

import lappieo.utils as ut

##################### DATASETS #####################

class ImagePathDataset(torch.utils.data.Dataset):
    
    def __init__(self, filenames, 
                        y, 
                        output_size, 
                        channels=None, 
                        array_transform=None, 
                        tensor_transform=None,
                        preload_tensor_transform=None,
                        load_to_memory=True):
        self.filenames = filenames
        self.y = y
        self.output_size = output_size

        self.channels = channels
        self.array_transform = array_transform
        self.tensor_transform = tensor_transform
        self.preload_tensor_transform = preload_tensor_transform
        self.mem_dataset = None

        if load_to_memory:
            self.mem_dataset = []
            print('Loading dataset to memory...')
            for i in tqdm(range(len(filenames))):
                self.mem_dataset.append(self.__readfile(i))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index): 
        if self.mem_dataset:
            X = self.mem_dataset[index]
        else:
            X = self.__readfile(index)

        if self.tensor_transform:
            X = self.tensor_transform(X)
        if self.y is not None:
            y = torch.as_tensor(self.y[index])
        else:
            y = torch.tensor(1)
        return X, y

    def __readfile(self, index):
        fname = self.filenames[index]
        img, _ = ut.read_raster_for_classification(fname)
        if self.channels:
            img = img[:,:,self.channels]

        if self.array_transform:
            img = self.array_transform(img)
        if img.shape[:2] != self.output_size:
            img = resize(img, self.output_size, order=0, preserve_range=True)
            
        img = img.astype(np.float)
        img = np.moveaxis(img,2,0)
        T = torch.as_tensor(img, dtype=torch.float)
        
        if self.preload_tensor_transform:
            T = self.preload_tensor_transform(T)
        
        return T
        
class ArrayDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = torch.tensor(self.X[index,:]).type(torch.float)
        
        y = torch.tensor(self.y[index])
        return X, y

class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        'Initialization'
        self.dataset = dataset
        self.transform = transform
        self.mem_dataset = []
        print('Loading dataset to memory...')
        for sample in tqdm(dataset):
            self.mem_dataset.append(sample)
        print('Done.')
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        X, y = self.mem_dataset[index]
        if self.transform:
            X = self.transform(X)
        return X, y

##################### PYTORCH LIGHTNING MODULES #####################

class _Sentinel2_ResNet50(nn.Module):
    def __init__(self, N_channels, N_classes, freeze_base=True):
        super().__init__()
        
        base_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False

        base_model.fc = nn.Linear(base_model.fc.in_features, N_classes) #requires_grad = True
        base_model.conv1 = nn.Conv2d(N_channels,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        
        self.model = base_model
        
    def forward(self, x): 
        return self.model(x)

class Sentinel2_ResNet50(nn.Module):
    def __init__(self, N_channels, N_classes, freeze_base=True, resnet_model='resnet50', pretrained=True):
        super().__init__()
        
        self.base_model = torch.hub.load('pytorch/vision:v0.6.0', resnet_model, pretrained=pretrained)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.h_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.base_model.conv1 = nn.Conv2d(N_channels,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        
        self.proj_head = nn.Sequential(
                        nn.Linear(self.h_dim, N_classes)
                        )
        
    def forward(self, x): 
        h = self.base_model(x)
        return self.proj_head(h)

    def base_forward(self, x):
        return self.base_model(x)

    def proj_forward(self, h):
        return self.proj_head(h)

class ICC_Sentinel2_ResNet50(nn.Module):
    def __init__(self, N_channels, N_classes, N_overcluster, resnet_model='resnet18', pretrained=True):
        super().__init__()
        self.N_classes = N_classes
        self.N_overcluster = N_overcluster
        
        self.base_model = torch.hub.load('pytorch/vision:v0.6.0', resnet_model, pretrained=pretrained)

        self.h_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.base_model.conv1 = nn.Conv2d(N_channels,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        
        self.headA = nn.Linear(self.h_dim, self.N_overcluster)
        self.headB = nn.Linear(self.h_dim, self.N_classes)
        
    def forward(self, x, head='B'): 
        h = self.base_model(x)

        if head == 'A':
            out = self.headA(h)
        elif head == 'B':
            out = self.headB(h)

        return out

    def base_forward(self, x):
        return self.base_model(x)

    def proj_forward(self, h, head='B'):
        if head == 'A':
            out = self.headA(h)
        elif head == 'B':
            out = self.headB(h)
        return out


##################### PYTORCH UTILITIES #####################

def multi_guess(x_batch, transforms, model, guesses=3):
    if isinstance(transforms, list):
        assert(len(transforms) == guesses)
    else:
        transforms = [transforms for _ in range(guesses)]

    Xlist = [transforms[i](x_batch) for i in range(guesses)]
    outputs = torch.stack([model(xb) for xb in Xlist],0)
    outputs = outputs.mean(dim=0)
    return outputs

def torch_predict(model, dataloader, gpu=True, n_guess=None, tf=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    y_true = []
    y_pred = []
    all_outputs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch

            x = x.to(device)
                
            if n_guess:
                outputs = multi_guess(x, tf, model, guesses=n_guess)
            else:
                outputs = model(x)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(dim=0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            outputs = outputs.detach().cpu()
            predicted = predicted.cpu()

            y_true.append(y.numpy())
            y_pred.append(predicted.numpy())
            all_outputs.append(outputs.numpy())
            
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)

    return y_true, y_pred, all_outputs

    

def intermediate_outputs(model, dataloader, gpu=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_outputs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch

            x = x.to(device)
            
            outputs = model.base_model(x)
            outputs = outputs.detach().cpu()

            all_outputs.append(outputs.numpy())
            
    all_outputs = np.concatenate(all_outputs, axis=0)

    return all_outputs
   
def load_dataset_to_memory(dataset):
    X = []
    y = []
    for i in tqdm(range(len(dataset))):
        xtemp, ytemp = dataset[i]
        X.append(xtemp)
        y.append(ytemp)
    return MemoryDataset(X, y)

##################### LOSS FUNCTIONS ###########################

import sys


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j

def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss
