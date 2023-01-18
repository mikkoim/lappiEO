# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:18:28 2020

@author: E1007914
"""

import glob
import os
import sys
import argparse

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import wandb

import src.classificationutils as cu
import dataset_stats


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str,
                    help="Folder with the training tiffs")

parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset for statistics")

parser.add_argument("--img_size", type=int, default=49,
                    help="all tifs are resized to this size")

parser.add_argument("--batch_size", type=int, default=128,
                    help="training batch size")

parser.add_argument("--epochs", type=int, default=10)

parser.add_argument("--learning_rate", type=float, default=1e-4,
                    help="training learning rate")

parser.add_argument("--resnet_model", type=str, required=False,
                    default="resnet18")

parser.add_argument("--load_dataset_to_memory", default=False, action="store_true")

parser.add_argument("--channels", type=str, default="[0,1,2,3,4,5,6,7,8,9,10]",
                    help="input image channels that are used")

parser.add_argument("--output_model_name", type=str, required=True,
                    help="name of the output model")

parser.add_argument("--n_clusters", type=int, default=5,
                    help="Ground truth cluster amount")

parser.add_argument("--n_overcluster", type=int, default=70,
                    help="Overclustering head number of classes")

parser.add_argument("--n_transforms", type=int, default=5,
                    help="Number of transforms performed on a sample")

parser.add_argument("--center_crop", type=int, default=9,
                    help="Center crop size, used for segmentation")
parser.add_argument("--random_crop", default=False, action="store_true")
parser.add_argument("--wandb_project", type=str, default='tests')

args = parser.parse_args()

def main(args):
    DATASET = args.dataset
    STATS0 = dataset_stats.stats[DATASET]
    STATS = {}

    FOLDER = args.folder
    OUTPUT_SIZE = (args.img_size, args.img_size)

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    LOAD_TO_MEMORY = args.load_dataset_to_memory

    CHANNELS = list(map(int, args.channels.strip('[]').split(',')))
    print("Channels: ", CHANNELS)

    MODEL_NAME = args.output_model_name

    N_CLUSTERS = args.n_clusters
    N_OVERCLUSTER = args.n_overcluster
    N_TRANSFORMS = args.n_transforms
    K = args.center_crop

    STATS['mean'] = STATS0['mean'][CHANNELS]
    STATS['std'] = STATS0['std'][CHANNELS]

    #%% Transforms
    ##

    center_crop = transforms.CenterCrop(args.center_crop)

    TF_PRELOAD = transforms.Compose([
                    transforms.Normalize(STATS['mean'], STATS['std'])
                    ])

    TF_AUG = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90)
                    ])

    cropmax = (args.center_crop+1)//2
    TF_CROP = lambda x: transforms.CenterCrop(np.random.randint(1,cropmax)*2+1)(x)
    if args.random_crop:
        TF_NOISE = lambda x: TF_CROP(TF_AUG(x))
    else:
        TF_NOISE = TF_AUG

    print("Nodata value: ", STATS0['nodata'])
    def remove_nodata(img):
        img[img==STATS0['nodata']] = 0
        return img
    array_transform = remove_nodata

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    #%% Process args
    fnames = glob.glob(os.path.abspath(os.path.join(FOLDER,'*.tif')))
    print("N files: ", len(fnames))
    N_channels = len(CHANNELS)

    dataset = cu.ImagePathDataset(fnames,
                                y = None,
                                output_size=OUTPUT_SIZE,
                                channels=CHANNELS,
                                array_transform=array_transform,
                                preload_tensor_transform=TF_PRELOAD,
                                load_to_memory=LOAD_TO_MEMORY)

    drop_last = lambda x: True if len(x)%BATCH_SIZE == 1 else False

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            drop_last=drop_last(dataset))

    #%%  Training
    net = cu.ICC_Sentinel2_ResNet50(N_channels=N_channels,
                                    N_classes=N_CLUSTERS,
                                    N_overcluster=N_OVERCLUSTER,
                                    resnet_model=args.resnet_model)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    hparams = {'model': net.__class__.__name__,
                'out_name': MODEL_NAME,
                'output_size': OUTPUT_SIZE,
                'N_channels': len(CHANNELS),
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE,
                'N_clusters': N_CLUSTERS,
                'N_overcluster': N_OVERCLUSTER,
                'N_transforms': N_TRANSFORMS,
                'Center crop': K,
                'epochs': EPOCHS,
                'aug': TF_NOISE}
    print(hparams)

    wandb.init(project=args.wandb_project,
       config=hparams)
    wandb.watch(net)

    step = 1

    for epoch in tqdm(range(EPOCHS)):
        print("Epoch:", epoch+1)
        for head in ['A', 'B']:
            print(head)
            for batch in tqdm(dataloader):
                net.train()
                optimizer.zero_grad()

                x, y = batch
                x = x.to(device)
                y = y.to(device)

                out = net.forward(center_crop(x), head=head)
                avg_loss = None

                for _ in range(N_TRANSFORMS):
                    out_tf = net.forward(center_crop(TF_NOISE(x)), head=head)

                    loss  = cu.IID_loss(F.softmax(out, dim=1), 
                                        F.softmax(out_tf, dim=1))

                    if avg_loss is None:
                        avg_loss = loss
                    else:
                        avg_loss += loss

                avg_loss /= N_TRANSFORMS

                if step % 10 == 0:
                    wandb.log({"loss":avg_loss.item(), "epoch":epoch})

                avg_loss.backward()
                optimizer.step()
                step += 1

    wandb.finish()
    torch.save(net.state_dict(), MODEL_NAME)

if __name__=='__main__':
    main(args)
    sys.exit()
