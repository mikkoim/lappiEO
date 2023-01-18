# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:37:34 2020

"""

import os
import wandb

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

import lappieo.utils as ut
import lappieo.classificationutils as cu
from lappieo import dataset_stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, required=True,
                    help="Location of the csv containing filenames and labels")
parser.add_argument("--test_csv", type=str, required=True,
                    help="location of the test dataset csv")
                    
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset for statistics")

parser.add_argument("--img_size", type=int, default=49,
                    help="all tifs are resized to this size")
parser.add_argument("--center_crop", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=64,
                    help="training batch size")

parser.add_argument("--epochs", type=int, default=10)

parser.add_argument("--learning_rate", type=float, default=1e-8,
                    help="training learning rate")
parser.add_argument("--load_dataset_to_memory", default=False, action="store_true")
parser.add_argument("--channels", type=str, default="[0,1,2,3,4,5,6,7,8,9,10]",
                    help="input image channels that are used")
parser.add_argument("--output_model_name", type=str, required=True,
                    help="name of the output model")
parser.add_argument("--resnet_model", type=str, required=False,
                    default="resnet50")
parser.add_argument("--freeze_base", default=False, action="store_true")
parser.add_argument("--random_crop", default=False, action="store_true")
parser.add_argument("--weighted", default=False, action="store_true")
parser.add_argument("--pretrained_model", type=str, default=None,
                    help="uses a pretrained model given as string")
parser.add_argument("--random_init_weights",  default=False, action="store_true")
parser.add_argument("--N_channels_source", type=int, default=None)
parser.add_argument("--N_classes_source", type=int, default=None)
parser.add_argument("--wandb_project", type=str, default='tests')

parser.add_argument("--pretrain_clusters", type=int, default=None)
parser.add_argument("--pretrain_overclusters", type=int, default=None)

args = parser.parse_args()

def main(args):
    #%% Change these if necessary
    DATASET = args.dataset
    STATS0 = dataset_stats.stats[DATASET]
    STATS = {}


    #%%
    TRAIN_CSV = args.train_csv
    TEST_CSV = args.test_csv

    OUTPUT_SIZE=(args.img_size,args.img_size)
    BATCH_SIZE = args.batch_size

    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    LOAD_TO_MEMORY = args.load_dataset_to_memory

    CHANNELS = list(map(int, args.channels.strip('[]').split(',')))
    print("Channels: ", CHANNELS)

    MODEL_NAME = args.output_model_name
    FREEZE_BASE = args.freeze_base
    RANDOM_CROP = args.random_crop
    RESNET_MODEL = args.resnet_model

    PRETRAINED_MODEL = args.pretrained_model
    N_CHANNELS_SOURCE = args.N_channels_source
    N_CLASSES_SOURCE = args.N_classes_source
    PROJECT = args.wandb_project

    STATS['mean'] = STATS0['mean'][CHANNELS]
    STATS['std'] = STATS0['std'][CHANNELS]
    
    TF_PRELOAD = transforms.Compose([
                    transforms.Normalize(STATS['mean'], STATS['std']),
                    transforms.CenterCrop(args.center_crop)
                    ])
        
    TF_TRAIN = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([transforms.GaussianBlur((3,3))],
                                        p=0.5)
                    ])
    cropmax = (args.center_crop+1)//2
    TF_CROP = lambda x: transforms.CenterCrop(np.random.randint(1,cropmax)*2+1)(x)
    if RANDOM_CROP:
        TTA_TF = lambda x: TF_CROP(TF_TRAIN(x))
    else:
        TTA_TF = TF_TRAIN
    
    print("Nodata value: ", STATS0['nodata'])
    def remove_nodata(img):
        img[img==STATS0['nodata']] = 0
        return img
    array_transform = remove_nodata

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    #%%

    fnames_train, labels_train = ut.read_fname_csv(TRAIN_CSV)
    fnames_test, labels_test = ut.read_fname_csv(TEST_CSV)


    y, le = ut.encode_labels(labels_train+labels_test)
    ut.print_label_counts(y, le)
    N_classes = len(le.classes_)
    print("N filenames: ", len(fnames_train+fnames_test))
    print("N classes: ", N_classes)
    y_train = le.transform(labels_train)
    y_test = le.transform(labels_test)

    if args.weighted:
        weights = 1./torch.Tensor(np.histogram(y_train, bins=len(le.classes_))[0])
        weights = weights.to(device)
    else:
        weights = None

    # Datasets and dataloaders
    print("Create dataloaders...")

    trainset = cu.ImagePathDataset(fnames_train, 
                                y_train, 
                                output_size=OUTPUT_SIZE,
                                channels=CHANNELS,
                                array_transform=array_transform,
                                preload_tensor_transform=TF_PRELOAD,
                                tensor_transform=TF_TRAIN,
                                load_to_memory=LOAD_TO_MEMORY)

    testset = cu.ImagePathDataset(fnames_test, 
                                y_test, 
                                output_size=OUTPUT_SIZE,
                                channels=CHANNELS,
                                array_transform=array_transform,
                                preload_tensor_transform=TF_PRELOAD,
                                load_to_memory=LOAD_TO_MEMORY)

    subset_ind = torch.randperm(len(trainset))[:len(testset)]
    train_subset = torch.utils.data.Subset(trainset, subset_ind)
    
    drop_last = lambda x: True if len(x)%BATCH_SIZE == 1 else False

    cpu_count = int(
            os.getenv("SLURM_CPUS_PER_TASK") or torch.multiprocessing.cpu_count()
        )

    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=BATCH_SIZE, 
                                                shuffle=True,
                                                drop_last=drop_last(trainset),
                                                num_workers=cpu_count)

    test_eval_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=BATCH_SIZE,
                                                drop_last=drop_last(testset),
                                                num_workers=cpu_count)
    train_eval_loader = torch.utils.data.DataLoader(train_subset,  
                                                batch_size=BATCH_SIZE,
                                                drop_last=drop_last(train_subset),
                                                num_workers=cpu_count)


    # Training

    model = cu.Sentinel2_ResNet50(len(CHANNELS), 
                                    N_classes, 
                                    freeze_base=FREEZE_BASE,
                                    resnet_model=RESNET_MODEL,
                                    pretrained= (not args.random_init_weights))

    if PRETRAINED_MODEL:
        print("Using pretrained model {}".format(PRETRAINED_MODEL))
        model = cu.Sentinel2_ResNet50(N_CHANNELS_SOURCE, 
                                N_CLASSES_SOURCE, 
                                freeze_base=FREEZE_BASE,
                                resnet_model=RESNET_MODEL)
        model.to(device)

        if args.pretrain_clusters:
            print('using unsupervised pretrain model')
            iic_model = cu.ICC_Sentinel2_ResNet50(len(CHANNELS),
                                    args.pretrain_clusters,
                                    args.pretrain_overclusters,
                                    args.resnet_model)

            iic_model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))
            model.base_model.load_state_dict(iic_model.base_model.state_dict())
        else:
            model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))

        if FREEZE_BASE:
            for param in model.parameters():
                param.requires_grad = False

        model.proj_head = torch.nn.Sequential(
                        torch.nn.Linear(model.h_dim, N_classes)
                        )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    hparams = {'model': model.__class__.__name__,
                'out_name': MODEL_NAME,
                'output_size': OUTPUT_SIZE,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE,
                'N_classes': N_classes,
                'N_channels': len(CHANNELS),
                'epochs': EPOCHS,
                'train_aug': TF_TRAIN}
    print(hparams)
    
    wandb.init(project=PROJECT, 
       config=hparams,
       name=args.output_model_name)
    wandb.watch(model)
    wandb.config.update(args)

    model.train()
    step = 1
    st_epoch = 0
    
    def validate(model, dataloader, n_guess=None, tf=None):
        y_true, y_pred, _ = cu.torch_predict(model, dataloader, n_guess=n_guess, tf=tf)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, f1

    for epoch in range(st_epoch, EPOCHS):
        print("Epoch: ", epoch+1)

        # Train
        for batch in tqdm(trainloader):
            optimizer.zero_grad()

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            if RANDOM_CROP:
                x = TF_CROP(x)
            out = model(x)

            loss = F.cross_entropy(out, y, weight=weights)

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                wandb.log({"loss":loss.item(), "epoch":epoch})
            step += 1
        
        # Validation
        acc_test, f1_test = validate(model, test_eval_loader)
        acc_test_multi, f1_test_multi = validate(model, test_eval_loader, n_guess=5, tf=TTA_TF)
        acc_train, f1_train = validate(model, train_eval_loader)
        
        wandb.log({"val acc":acc_test, 
                    "val f1":f1_test, 
                    "train acc":acc_train, 
                    "train f1":f1_train,
                    "val acc multi":acc_test_multi,
                    "val f1 multi": f1_test_multi,
                    "epoch":epoch})
        st_epoch += 1
        
    wandb.finish()
    torch.save(model.state_dict(), MODEL_NAME)

if __name__=='__main__':
    main(args)
    exit()
