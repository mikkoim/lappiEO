# -*- coding: utf-8 -*-
"""
"""

import numpy as np

import wandb

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F
from torchvision import transforms

import src.utils as ut
import src.classificationutils as cu
import dataset_stats

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
parser.add_argument("--batch_size", type=int, default=64,
                    help="training batch size")

parser.add_argument("--epochs", type=int, default=10)

parser.add_argument("--learning_rate", type=float, default=1e-8,
                    help="training learning rate")
parser.add_argument("--use_gpu", default=False, action="store_true")
parser.add_argument("--load_dataset_to_memory", default=False, action="store_true")
parser.add_argument("--channels", type=str, default="[0,1,2,3,4,5,6,7,8,9,10]",
                    help="input image channels that are used")
parser.add_argument("--output_model_name", type=str, required=True,
                    help="name of the output model")

parser.add_argument("--freeze_base", default=False, action="store_true")
parser.add_argument("--pretrained_model", type=str, default=None,
                    help="uses a pretrained model given as string")
parser.add_argument("--N_channels_source", type=int, default=None)
parser.add_argument("--N_classes_source", type=int, default=None)
parser.add_argument("--wandb_project", type=str, default='tests')

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
    GPU = args.use_gpu
    LOAD_TO_MEMORY = args.load_dataset_to_memory

    CHANNELS = list(map(int, args.channels.strip('[]').split(',')))
    print("Channels: ", CHANNELS)

    MODEL_NAME = args.output_model_name
    FREEZE_BASE = args.freeze_base

    PRETRAINED_MODEL = args.pretrained_model
    N_CHANNELS_SOURCE = args.N_channels_source
    N_CLASSES_SOURCE = args.N_classes_source

    STATS['mean'] = STATS0['mean'][CHANNELS]
    STATS['std'] = STATS0['std'][CHANNELS]


    TF_PRELOAD = transforms.Normalize(STATS['mean'], STATS['std'])
    TF_TRAIN = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([transforms.GaussianBlur((3,3))],
                                        p=0.5)
                    ])

    print("Nodata value: ", STATS0['nodata'])
    def remove_nodata(img):
        img[img==STATS0['nodata']] = 0
        return img
    array_transform = remove_nodata

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
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    trainloader_h = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    test_eval_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    train_eval_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE)


    # Training

    model = cu.Sentinel2_ResNet50(len(CHANNELS), N_classes, freeze_base=FREEZE_BASE)

    if PRETRAINED_MODEL:
        print("Using pretrained model {}".format(PRETRAINED_MODEL))
        model = cu.Sentinel2_ResNet50(N_CHANNELS_SOURCE,
                                N_CLASSES_SOURCE, freeze_base=FREEZE_BASE)
        if GPU:
            model.to('cuda')

        model.load_state_dict(torch.load(PRETRAINED_MODEL))

        if FREEZE_BASE:
            for param in model.parameters():
                param.requires_grad = False

        model.proj_head = torch.nn.Sequential(
                        torch.nn.Linear(model.h_dim, N_classes)
                        )

    if GPU:
        print("Using GPU")
        model.to('cuda')
    else:
        print("Using CPU")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    hparams = {'model': model.__class__.__name__,
            'out_name': MODEL_NAME,
                'output_size': OUTPUT_SIZE,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE,
                'N_classes': N_classes,
                'N_channels': len(CHANNELS),
                'epochs': EPOCHS}
    print(hparams)

    wandb.init(project="tests", 
       config=hparams)
    wandb.watch(model)

    model.train()
    step = 1
    st_epoch = 0

    def validate(model, dataloader):
        y_true, y_pred, _ = cu.torch_predict(model, dataloader, gpu=GPU)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, f1

    def softCrossEntropy(input, target):
        logprobs = F.log_softmax(input, dim=1)
        return  -(F.softmax(target, dim=1) * logprobs).sum(dim=1).mean()

    def y_to_hot(y, c):
        bn = len(y)
        y = y.view(-1,1)
        y_hot = torch.zeros(bn,c)
        y_hot.scatter_(1,y,1)
        return y_hot

    for epoch in range(st_epoch, EPOCHS):
        print("Epoch: ", epoch+1)

        # Train
        th_iter = iter(trainloader_h)
        for batch in tqdm(trainloader):
            batch_h = next(th_iter)
            optimizer.zero_grad()

            x, y = batch
            x_h, y_h = batch_h

            y = y_to_hot(y, N_classes)
            y_h = y_to_hot(y_h, N_classes)
            lamb = torch.rand(len(y),1)

            if GPU:
                x = x.to('cuda')
                y = y.to('cuda')

                x_h = x_h.to('cuda')
                y_h = y_h.to('cuda')
                lamb = lamb.to('cuda')

            h = model.base_forward(x)
            h_h = model.base_forward(x_h)

            h_p = lamb*h + (1-lamb)*h_h
            y_p = lamb*y + (1-lamb)*y_h

            out = model.proj_forward(h_p)
            loss = softCrossEntropy(out, y_p)

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                wandb.log({"loss":loss.item(), "epoch":epoch})
            step += 1
        
        # Validation
        acc_test, f1_test = validate(model, test_eval_loader)
        acc_train, f1_train = validate(model, train_eval_loader)

        wandb.log({"val acc":acc_test, 
                    "val f1":f1_test, 
                    "train acc":acc_train, 
                    "train f1":f1_train,
                    "epoch":epoch})
        st_epoch += 1
        
    wandb.finish()
    torch.save(model.state_dict(), MODEL_NAME)

if __name__=='__main__':
    main(args)
    exit()