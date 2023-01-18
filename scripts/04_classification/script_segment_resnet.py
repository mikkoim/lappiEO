# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:07:54 2020

@author: E1007914
"""

import numpy as np
import matplotlib.pyplot as plt

import src.classificationutils as cu
import src.utils as ut
import dataset_stats

from skimage.color import label2rgb

import skimage.io as sio
import os

import torch
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_fname", type=str, required=True,
                    help="Location of segmentation weights")
parser.add_argument("--tif_folder", type=str, required=True,
                    help="Location of georaster tiff")

parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset for statistics")

parser.add_argument("--n_channels", type=int, required=True,
                    help="number of channels in seg network")

parser.add_argument("--n_classes", type=int, required=True,
                    help="number of classes in seg network")

parser.add_argument("--block_size", type=int, required=True,
                    help="size of in-memory segmented block - depends on gpu memory")

parser.add_argument("--input_size", type=int, required=True,
                    help="size of lookaround pixels")

parser.add_argument("--out_folder", type=str, required=True)
parser.add_argument("--out_appendix", type=str, required=True)


args = parser.parse_args()

def main(args):
    DATASET = args.dataset
    STATS = dataset_stats.stats[DATASET]

    MODEL_FNAME = args.model_fname
    TIF_FOLDER = args.tif_folder

    N_CHANNELS = args.n_channels
    N_CLASSES = args.n_classes

    BLOCK_SIZE = args.block_size
    INPUT_SIZE = args.input_size
    OUT_FOLDER = args.out_folder
    OUT_APPENDIX = args.out_appendix

    def discrete_cmap(N, base_cmap=None):

        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        return color_list

    colors = discrete_cmap(N_CLASSES, 'tab20')

    # Transforms
    tf = transforms.Compose([
                    transforms.Normalize(STATS['mean'], STATS['std'])
                    ])


    # Model
    model = cu.Sentinel2_ResNet50(N_CHANNELS, N_CLASSES, resnet_model='resnet18')
    model.to('cuda')
    model.load_state_dict(torch.load(MODEL_FNAME))

    model.eval()

    # Loading
    filenames = []
    for fname in os.listdir(TIF_FOLDER):
        if fname.endswith('.tif'):
            filenames.append(
                os.path.abspath(
                    os.path.join(TIF_FOLDER,fname)))
    
    print(f'{len(filenames)} tif files in {TIF_FOLDER}')

    for TIF_FNAME in filenames:
        print(TIF_FNAME)
        array, src = ut.read_raster_for_classification(TIF_FNAME)
        print("Original array shape: ", array.shape)

        print("Resetting nodata value: ", STATS['nodata'])
        array[array==STATS['nodata']] = 0
            
        T = torch.Tensor(np.moveaxis(array.astype(np.int16), 2,0))
        T = tf(T)
        print("Tensor shape: ", T.shape)

        # Segmentation
        segmented, outputimg = ut.segment_large_tensor(T, 
                                                model, 
                                                classes=N_CLASSES,
                                                k=INPUT_SIZE, 
                                                block_size=BLOCK_SIZE)

        print("Segmented shape: ", segmented.shape)

        S_img = label2rgb(segmented, colors=colors)
        outimg = torch.nn.functional.softmax(outputimg, dim=0)
        onp = outimg.cpu().numpy()
        onp = np.int16(np.around(onp,2)*100)

        # Saving
        fname = os.path.splitext(os.path.basename(TIF_FNAME))[0]
        outname = os.path.join(OUT_FOLDER, f"{fname}_{OUT_APPENDIX}")
        sio.imsave(outname+'.png',S_img)
        ut.write_classified_raster(outname, src, segmented)
        ut.write_float_raster(outname + 'scores', src, onp, N_CLASSES)

if __name__=='__main__':
    main(args)
    exit()