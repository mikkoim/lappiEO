## RF classification

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:07:54 2020

@author: E1007914
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import src.classificationutils as cu
import src.utils as ut
import dataset_stats

from skimage.color import label2rgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import skimage.io as sio

import torch
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, required=True,
                    help="Location of the csv containing filenames and labels")
parser.add_argument("--test_csv", type=str, required=True,
                    help="location of the test dataset csv")
parser.add_argument("--tif_folder", type=str, required=True,
                    help="georaster tiff folder to be classified")
                    
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset for statistics")

parser.add_argument("--img_size", type=int, default=49,
                    help="all tifs are resized to this size")

parser.add_argument("--channels", type=str, default=None,
                    help="input image channels that are used")

parser.add_argument("--out_folder", type=str, required=True)
parser.add_argument("--out_appendix", type=str, required=True)

args = parser.parse_args()


TRAIN_CSV = args.train_csv
TEST_CSV = args.test_csv

CHANNELS = list(map(int, args.channels.strip('[]').split(',')))
OUTPUT_SIZE=(args.img_size,args.img_size)

DATASET = args.dataset
STATS = dataset_stats.stats[DATASET]

TIF_FOLDER = args.tif_folder

OUT_FOLDER = args.out_folder
OUT_APPENDIX = args.out_appendix

# Extract center pixel as features
class IdentityModel(torch.nn.Module):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
    def forward(self, x):
        if self.transform:
            x = self.transform(x)
        return x.squeeze()

feature_model = IdentityModel(transform=transforms.CenterCrop(1))

# Create dataloaders
fnames_train, labels_train = ut.read_fname_csv(TRAIN_CSV)
fnames_test, labels_test = ut.read_fname_csv(TEST_CSV)
y, le = ut.encode_labels(labels_train+labels_test)
N_CLASSES = len(le.classes_)
print(N_CLASSES)

y_train = le.transform(labels_train)
y_test = le.transform(labels_test)

print("Trainset:")
ut.print_label_counts(y_train,le)
print("Testset:")
ut.print_label_counts(y_test,le)

trainset = cu.ImagePathDataset(fnames_train, 
                              y_train, 
                              output_size=OUTPUT_SIZE,
                              channels=CHANNELS)

testset = cu.ImagePathDataset(fnames_test, 
                              y_test, 
                              output_size=OUTPUT_SIZE,
                              channels=CHANNELS)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128)
testloader = torch.utils.data.DataLoader(testset, batch_size=128)


y_true_train, y_pred_train, X_train = cu.torch_predict(feature_model, trainloader, gpu=False)
y_true_test, y_pred_test, X_test = cu.torch_predict(feature_model, testloader, gpu=False)


# Train classifier
clf = RandomForestClassifier(n_estimators=100,
                             n_jobs=-1,
                             verbose=2)
print(y_true_train[:5])
clf.classes_ = le.classes_
clf.fit(X_train, y_true_train)

# Evaluation
y_pred_ref = clf.predict(X_test)
print(clf)
print(classification_report(le.inverse_transform(y_true_test), 
                            le.inverse_transform(y_pred_ref),
                            zero_division=0))

# Colors
def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    return color_list

colors = discrete_cmap(N_CLASSES, 'tab20')


# Actual raster classification
filenames = []
for fname in os.listdir(TIF_FOLDER):
    if fname.endswith('.tif'):
        filenames.append(
            os.path.abspath(
                os.path.join(TIF_FOLDER,fname)))

for TIF_FNAME in filenames:
    A, src = ut.read_raster_for_classification(TIF_FNAME)
    print("Original array shape: ", A.shape)

    print("Resetting nodata value: ", STATS['nodata'])
    A[A==STATS['nodata']] = 0

    ny, nx, chan = A.shape
    a = A.reshape(ny*nx, chan)

    C0 = clf.predict_proba(a)

    C = np.zeros((C0.shape[0], N_CLASSES)) #Trainset might not have all classes
    C[:,clf.classes_] = C0

    C = C.reshape(ny,nx,-1)
    C = np.moveaxis(C,2,0)

    segmented = np.argmax(C,0)

    S_img = label2rgb(segmented, colors=colors)
    onp = np.int16(np.around(C,2)*100)

    # Saving
    fname = os.path.splitext(os.path.basename(TIF_FNAME))[0]
    outname = os.path.join(OUT_FOLDER, f"{fname}_{OUT_APPENDIX}")
    sio.imsave(outname+'.png',S_img)
    ut.write_classified_raster(outname, src, segmented)
    ut.write_float_raster(outname + 'scores', src, onp, N_CLASSES)