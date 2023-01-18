import os
import re
import yaml
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

import skimage.io
from skimage.color import label2rgb
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import wandb

import src.utils as ut
import src.classificationutils as cu
import src.highlevelutils as hu
import dataset_stats

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True,
                    help="Location of the config file")
parser.add_argument("--file_dir", type=str)
parser.add_argument("--label_dir", type=str)
parser.add_argument("--train_csv", type=str)
parser.add_argument("--test_csv", type=str)

args = parser.parse_args()

class SegmentationDataset(torch.utils.data.Dataset):
    
    def __init__(self, fname_label_csv,
                        load_transform=None,
                        transform=None,
                        load_to_memory=True,
                        as_tensor = True):
        
        self.fnames, self.labels = ut.read_fname_csv(fname_label_csv)
        
        self.load_transform = load_transform
        self.transform= transform
        self.mem_dataset = None
        self.as_tensor = as_tensor
        
        if load_to_memory:
            self.mem_dataset = []
            print('Loading dataset to memory...')
            for i in tqdm(range(len(self.labels))):
                self.mem_dataset.append(self.__readfile(i))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index): 
        if self.mem_dataset:
            img, mask = self.mem_dataset[index]
        else:
            img, mask = self.__readfile(index)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            
        if self.as_tensor:
            img = img.astype(float)
            mask = mask.astype(float)
            img = ToTensor()(img).float()
            mask = ToTensor()(mask).squeeze().long()
        return img, mask

    def __readfile(self, index):
        
        filepath = self.fnames[index]
        labelpath = self.labels[index]
        
        img = skimage.io.imread(filepath) 
        mask = skimage.io.imread(labelpath)
        
        if self.load_transform:
            aug = self.load_transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        
        return img, mask

def remove_nodata(img):
    img[img==STATS['nodata']] = 0
    return img

class RemoveNoData(ImageOnlyTransform):
    def apply(self, img, **params):
        return remove_nodata(img)

def tensor2rgb(T, i):
    if i < 0:
        img = T.detach().cpu().numpy()[:,[4,3,2],:,:]
        img = np.moveaxis(img / np.expand_dims(np.expand_dims(img.max(-1).max(-1), 2),2),1,3)
    else:
        img = T.detach().cpu().numpy()[i,[4,3,2],:,:]
        img = np.moveaxis(img,0,2) / img.max(-1).max(-1)
    return img
    
def aug_batch(batch, tf):
    for i in range(batch.shape[0]):
        x = batch[i,::].permute(1,2,0).cpu().numpy()
        xhat = torch.Tensor(tf(image=x)['image']).permute(2,0,1)

        if i == 0:
            batch_t = torch.zeros(batch.shape[0], xhat.shape[0], xhat.shape[1], xhat.shape[2])
            
        batch_t[i,::] = xhat
    return batch_t
    
def dice_loss(prediction, target):
    assert(np.all(np.unique(target.detach().cpu()) == np.array([0,1])))
    eps = 1e-8

    A = prediction.view(-1)
    B = target.view(-1)
    intersection = (A * B).sum()
    
    return 1-((2. * intersection) / (A.sum() + B.sum() + eps))

one_hot = lambda x: F.one_hot(x, num_classes=36)
dice_eval = lambda x,y: dice_loss(one_hot( torch.Tensor(x).long() ), 
                                  one_hot( torch.Tensor(y).long() )).item()

def predict(dataloader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    y_true = []
    y_pred = []
    y_score = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x = x.to(device)
            
            outputs = model(x)
            
            predicted = torch.argmax(outputs, dim=1)
            
            outputs = outputs.detach().cpu()
            predicted = predicted.cpu()

            y_true.append(y.numpy())
            y_pred.append(predicted.numpy())
            y_score.append(outputs.numpy())
            
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    return y_true, y_pred, y_score


def predict_center(dataloader, model, transform):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
        
            x = aug_batch(x, transform)
            x = x.to(device)
            
            out = model(x)
            
            predicted = torch.argmax(out, dim=1)
            predicted = torchvision.transforms.CenterCrop(1)(predicted).squeeze()

            y_true.append(y.numpy())
            y_pred.append(predicted.cpu().numpy())
            
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    return y_true, y_pred

###########################################################################

with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        
pprint(config)

### Segmentation train test split
print("Creating segmentation dataset...")
file_dir = args.file_dir
label_dir = args.label_dir
out_fname = config['csv_out_name']
subset = config['subset']

labels = []
files = []

for file in os.listdir(args.label_dir):
    if file.endswith('.tif'):
        f_ind = re.search(r'\d+', file).group()
        labels.append(os.path.abspath(os.path.join(args.label_dir,f'{f_ind}_segtrain.tif')))
        #scores.append(os.path.abspath(os.path.join(label_dir,f'{f_ind}_segtrainscores.tif')))
        files.append(os.path.abspath(os.path.join(args.file_dir,f'{f_ind}.tif')))
if subset:
    labels = labels[:config['subset']]
    files = files[:config['subset']]
    
fname_df = pd.DataFrame(list(zip(files,labels)), columns=['fname', 'label'])

np.random.seed(666)
train_ind = np.random.rand(len(fname_df)) < 0.9
train_df = fname_df[train_ind]
test_df = fname_df[~train_ind]

train_df.to_csv(f'{out_fname}_train.csv', index=False)
test_df.to_csv(f'{out_fname}_test.csv', index=False)


### Segmentation datasets and loaders
print("Loading segmentation dataset...")
STATS = dataset_stats.stats[config['dataset']]

tf_load = A.Compose(
    [
        RemoveNoData(),
        A.CenterCrop(49,49)
    ]
)

tf_train = A.Compose(
    [
        A.ToFloat(max_value=2**16),
        A.Resize(config['resize'], config['resize']),
        A.FromFloat(max_value=2**16),
        A.Normalize(mean=STATS['mean'],
                   std=STATS['std'],
                   max_pixel_value=1),
        A.Flip()
    ]
)

tf_test = A.Compose(
    [
        A.ToFloat(max_value=2**16),
        A.Resize(config['resize'], config['resize']),
        A.FromFloat(max_value=2**16),
        A.Normalize(mean=STATS['mean'],
                   std=STATS['std'],
                   max_pixel_value=1)
    ]
)

trainset = SegmentationDataset(f'{out_fname}_train.csv',
                        load_transform=tf_load,
                        transform=tf_train,
                        load_to_memory=True,
                        as_tensor=True)

testset = SegmentationDataset(f'{out_fname}_test.csv',
                        load_transform=tf_load,
                        transform=tf_test,
                        load_to_memory=True,
                        as_tensor=True)

subset_ind = torch.randperm(len(trainset))[:len(testset)]
train_subset = torch.utils.data.Subset(trainset, subset_ind)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'])
train_eval_loader = torch.utils.data.DataLoader(train_subset, batch_size=config['batch_size'])


## Classification dataset loading
print("Loading classification dataset...")
fnames_train, labels_train = ut.read_fname_csv(args.train_csv)
fnames_test, labels_test = ut.read_fname_csv(args.test_csv)


y, le = ut.encode_labels(labels_train+labels_test)
ut.print_label_counts(y, le)
N_classes = len(le.classes_)
print("N filenames: ", len(fnames_train+fnames_test))
print("N classes: ", N_classes)
y_train = le.transform(labels_train)
y_test = le.transform(labels_test)


classificationset = cu.ImagePathDataset(fnames_test, 
                            y_test, 
                            output_size=(49,49),
                            channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
                            array_transform=None,
                            preload_tensor_transform=None,
                            load_to_memory=True)

classificationloader = torch.utils.data.DataLoader(classificationset, batch_size=config['batch_size'],shuffle=False)

# Training
segmentation_array = lambda x,i: x[i,::].detach().cpu().squeeze().numpy()
print("Start training...")
wandb.init(project=config['wandb_project'], 
   config=config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=14, out_channels=36, init_features=config['init_features'])
model = model.to(device)

wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

# Visual evaluation test image
xb_val, yb_val = next(iter(testloader))
xb_val = xb_val.to(device)

step = 0
for epoch in range(config['epochs']):
    print("Epoch: ", epoch+1)

    # Train
    for batch in tqdm(trainloader):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        y = y.to(device)
        
        out = model(x)
        
        dice = dice_loss(out, one_hot(y))
        ce = F.cross_entropy(out, y)
        loss = ce

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            wandb.log({"dice loss": dice.item(), 
                       "cross entropy loss": ce.item(), 
                       "epoch": epoch})
            print(f'loss: {dice.item()}')
        step += 1
        
    # Validation
    y_true_test, y_pred_test, _ = predict(testloader, model)
    y_true_train, y_pred_train, _ = predict(train_eval_loader, model)

    dice_test = dice_eval(y_pred_test, y_true_test)
    dice_train = dice_eval(y_pred_train, y_true_train)
    f1_test = f1_score(y_true_test.ravel(), y_pred_test.ravel(), average='macro')
    f1_train = f1_score(y_true_train.ravel(), y_pred_train.ravel(), average='macro')
    
    # Classification
    y_true_c, y_pred_c = predict_center(classificationloader, model, A.Compose([tf_load, tf_test]))
    acc_c = accuracy_score(y_true_c, y_pred_c)
    f1_c = f1_score(y_true_c, y_pred_c, average='macro', zero_division=0)
    print(f'acc: {acc_c}\nf1: {f1_c}')
    
    wandb.log({"classification acc":acc_c, 
                "classification f1":f1_c, 
               "seg f1 test":f1_test,
               "seg f1 train":f1_train,
                "seg dice test":dice_test,
                "seg dice train":dice_train,
                "epoch":epoch})
    
    model.eval()
    with torch.no_grad():
        out_val = model(xb_val)
        S_val = torch.argmax(out_val, dim=1)
        
        for seg_i in range(2):
            mask_img = wandb.Image(tensor2rgb(xb_val,seg_i), masks={
                                  "ground_truth": { "mask_data": segmentation_array(yb_val,seg_i) },
                                  "predictions":  { "mask_data": segmentation_array(S_val,seg_i)}
                                    })
            wandb.log({f'test {seg_i}': mask_img})
    
torch.save(model.state_dict(), config['output_name'])
wandb.finish()
