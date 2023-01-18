# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:37:34 2020

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

parser.add_argument("--batch_size", type=int, default=128,
                    help="training batch size")

parser.add_argument("--epochs", type=int, default=10)

parser.add_argument("--learning_rate", type=float, default=1e-4,
                    help="training learning rate")

parser.add_argument("--use_gpu", default=False, action="store_true")
parser.add_argument("--load_dataset_to_memory", default=False, action="store_true")

parser.add_argument("--channels", type=str, default="[0,1,2,3,4,5,6,7,8,9,10]",
                    help="input image channels that are used")

parser.add_argument("--pretrained_model", type=str, default=None,
                    help="uses a pretrained model given as string")

parser.add_argument("--output_model_name", type=str, required=True,
                    help="name of the output model")

parser.add_argument("--freeze_base", default=False, action="store_true")

parser.add_argument("--n_clusters", type=int, default=5,
                    help="Ground truth cluster amount")

parser.add_argument("--n_overcluster", type=int, default=70,
                    help="Overclustering head number of classes")

parser.add_argument("--center_crop", type=int, default=9,
                    help="Center crop size, used for segmentation")
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

    N_CLUSTERS = args.n_clusters
    N_OVERCLUSTER = args.n_overcluster
    K = args.center_crop

    STATS['mean'] = STATS0['mean'][CHANNELS]
    STATS['std'] = STATS0['std'][CHANNELS]


    TF_TRAIN = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()
                    ])

    tf = transforms.Compose([
                    transforms.CenterCrop(K),
                    transforms.Normalize(STATS['mean'], STATS['std'])
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
                                load_to_memory=LOAD_TO_MEMORY,
                                array_transform=array_transform,
                                tensor_transform=TF_TRAIN)

    testset = cu.ImagePathDataset(fnames_test, 
                                y_test, 
                                output_size=OUTPUT_SIZE,
                                channels=CHANNELS,
                                load_to_memory=LOAD_TO_MEMORY,
                                array_transform=array_transform)

    subset_ind = torch.randperm(len(trainset))[:len(testset)]
    train_subset = torch.utils.data.Subset(trainset, subset_ind)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    eval_loader_test = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
    eval_loader_train = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE)

    # Training
    model = cu.ICC_Sentinel2_ResNet50(len(CHANNELS),
                                        N_CLUSTERS,
                                        N_OVERCLUSTER,
                                        [tf, tf])

    if GPU:
        model.to('cuda')

    model.load_state_dict(torch.load(PRETRAINED_MODEL))

    if FREEZE_BASE:
        for param in model.parameters():
            param.requires_grad = False

    model.headB = torch.nn.Linear(model.headB.in_features, N_classes)

    if GPU:
        print("Using GPU")
        model.to('cuda')
    else:
        print("Using CPU")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(optimizer.__class__.__name__)

    hparams = {'model': model.__class__.__name__,
                'out_name': MODEL_NAME,
                'output_size': OUTPUT_SIZE,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE,
                'N_classes': N_classes,
                'N_channels': len(CHANNELS),
                'epochs': EPOCHS,

                'N_clusters': N_CLUSTERS,
                'N_overcluster': N_OVERCLUSTER,
                'Center crop': K}
    print(hparams)

    wandb.init(project=args.wandb_project, 
       config=hparams)
    wandb.watch(model)

    def validate(model, dataloader):
        y_true, y_pred, _ = cu.torch_predict(model, dataloader, gpu=GPU)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, f1

    model.train()
    step = 1
    st_epoch = 0

    for epoch in range(st_epoch, EPOCHS):
        print("Epoch: ", epoch+1)

        # Train
        for batch in tqdm(trainloader):
            optimizer.zero_grad()

            x, y = batch
            if GPU:
                x = x.to('cuda')
                y = y.to('cuda')
            out = model(x)

            loss = F.cross_entropy(out, y)

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                wandb.log({"loss":loss.item(), "epoch":epoch})
                
            step += 1
        
        # Validation
        acc_test, f1_test = validate(model, eval_loader_test)
        acc_train, f1_train = validate(model, eval_loader_train)

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