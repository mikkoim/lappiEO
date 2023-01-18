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

import skimage.io as sio

import torch
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_fname", type=str, required=True,
                    help="Location of segmentation weights")
parser.add_argument("--tif_folder", type=str, required=True,
                    help="georaster tiff folder to be classified")

parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset for statistics")

parser.add_argument("--n_cluster", type=int, required=True,
                    help="number of clusters in seg network")
parser.add_argument("--n_overcluster", type=int, required=True,
                    help="number of overclusters in seg network")
parser.add_argument("--n_channels", type=int, required=True,
                    help="number of channels in seg network")
parser.add_argument("--use_head_a", default=False, action="store_true")

parser.add_argument("--block_size", type=int, required=True,
                    help="size of in-memory segmented block - depends on gpu memory")

parser.add_argument("--center_crop", type=int, required=True,
                    help="size of lookaround pixels")

parser.add_argument("--out_folder", type=str, required=True)
parser.add_argument("--out_appendix", type=str, required=True)


args = parser.parse_args()

def main(args):
    DATASET = args.dataset
    STATS = dataset_stats.stats[DATASET]

    MODEL_FNAME = args.model_fname
    TIF_FOLDER = args.tif_folder

    N_CLUSTER = args.n_cluster
    N_OVERCLUSTER = args.n_overcluster
    N_CHANNELS = args.n_channels
    USE_HEAD_A = args.use_head_a

    BLOCK_SIZE = args.block_size
    CENTER_CROP = args.center_crop
    OUT_FOLDER = args.out_folder
    OUT_APPENDIX = args.out_appendix

    def discrete_cmap(N, base_cmap=None):

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        return color_list

    colors = discrete_cmap(N_CLUSTER, 'tab20')

    tf = transforms.Compose([
                    transforms.Normalize(STATS['mean'], STATS['std'])
                    ])

    IIC_cluster = cu.ICC_Sentinel2_ResNet50(N_CHANNELS, N_CLUSTER, N_OVERCLUSTER, 'resnet18')
    IIC_cluster.to('cuda')
    IIC_cluster.load_state_dict(torch.load(MODEL_FNAME))

    output_classes = N_CLUSTER
    if USE_HEAD_A:
        class ModelAWrapper(torch.nn.Module):
            def __init__(self, orig_model):
                super().__init__()
                self.orig_model = orig_model
                
            def forward(self, x): 
                return self.orig_model(x, head='A')

        IIC_cluster = ModelAWrapper(IIC_cluster)
        output_classes = N_OVERCLUSTER

    IIC_cluster.eval()

    filenames = []
    for fname in os.listdir(TIF_FOLDER):
        if fname.endswith('.tif'):
            filenames.append(
                os.path.abspath(
                    os.path.join(TIF_FOLDER,fname)))

    for TIF_FNAME in filenames:
        # Loading
        array, src = ut.read_raster_for_classification(TIF_FNAME)
        print("Original array shape: ", array.shape)

        print("Resetting nodata value: ", STATS['nodata'])
        array[array==STATS['nodata']] = 0

        T = torch.Tensor(np.moveaxis(array.astype(np.int16), 2,0))
        T = tf(T)
        print("Tensor shape: ", T.shape)

        # Segmentation
        segmented, _ = ut.segment_large_tensor(T, 
                                            IIC_cluster,
                                            classes=output_classes,
                                            k=CENTER_CROP,
                                            block_size=BLOCK_SIZE)

        print("Segmented shape: ", segmented.shape)

        S_img = label2rgb(segmented, colors=colors)

        # Saving
        fname = os.path.splitext(os.path.basename(TIF_FNAME))[0]
        outname = os.path.join(OUT_FOLDER, f"{fname}_{OUT_APPENDIX}")
        sio.imsave(outname+'.png',S_img)
        ut.write_classified_raster(outname, src, segmented)

if __name__=='__main__':
    main(args)
    exit()