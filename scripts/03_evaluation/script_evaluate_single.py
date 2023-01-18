import argparse
import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score,
                             f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize
from torchvision import transforms

import dataset_stats
import src.evalutils as eutil

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", type=str, required=True,
                    help="Location of the csv containing filenames and labels")

parser.add_argument("--test_csv", type=str, required=True,
                    help="location of the test dataset csv")

parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset for statistics")

parser.add_argument("--model", type=str, default=None,
                    help="uses a pretrained model given as string")

parser.add_argument("--img_size", type=int, default=49,
                    help="all tifs are resized to this size")
parser.add_argument("--center_crop", type=int, required=True)

parser.add_argument("--batch_size", type=int, default=64,
                    help="training batch size")

parser.add_argument("--load_dataset_to_memory", default=False, action="store_true")
parser.add_argument("--crop", default=False, action="store_true")
parser.add_argument("--resnet_model", type=str, required=False,
                    default="resnet50")
parser.add_argument("--channels", type=str, default=None,
                    help="input image channels that are used")

parser.add_argument("--N_classes_source", type=int, default=None)
parser.add_argument("--tta_guesses", type=int, default=None)

args = parser.parse_args()

DATASET = args.dataset
STATS0 = dataset_stats.stats[DATASET]
STATS = {}
OUTPUT_SIZE=(args.img_size,args.img_size)

CHANNELS = list(map(int, args.channels.strip('[]').split(',')))
print("Channels: ", CHANNELS)

STATS['mean'] = STATS0['mean'][CHANNELS]
STATS['std'] = STATS0['std'][CHANNELS]

N_classes = args.N_classes_source

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

def remove_nodata(img):
    img[img==STATS0['nodata']] = 0
    return img

array_transform = remove_nodata

TF_PRELOAD = transforms.Compose([
                transforms.Normalize(STATS['mean'], STATS['std']),
                transforms.CenterCrop(args.center_crop)
                ])
TF_NOCROP = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()
                    ])

cropmax = (args.center_crop+1)//2
crop_scales = np.ceil(np.linspace(0,cropmax-1,5))*2+1
if args.crop:
    TF_TTA = []
    for ti in range(args.tta_guesses):
        TF_TTA.append(transforms.Compose([
            TF_NOCROP,
            transforms.CenterCrop(crop_scales[ti])
        ]))
else:
    TF_TTA = TF_NOCROP

LABEL_MAPS = [lambda x: x[:1], lambda x: x[:2], lambda x: x]

METRICS = {'top 1 accuracy':    lambda y_true, y_test, y_scores: accuracy_score(y_true, y_test),
           'top 3 accuracy':    lambda y_true, y_test, y_scores: eutil.top_k_accuracy_score(y_true, y_scores, k=3),
           'top 5 accuracy':    lambda y_true, y_test, y_scores: eutil.top_k_accuracy_score(y_true, y_scores, k=5),
           'f1 macro':          lambda y_true, y_test, y_scores: f1_score(y_true, y_test, average='macro', zero_division=0),
           'f1 weighted':       lambda y_true, y_test, y_scores: f1_score(y_true, y_test, average='weighted', zero_division=0),
           'precision macro':   lambda y_true, y_test, y_scores: precision_score(y_true, y_test, average='macro', zero_division=0),
           'precision weighted':lambda y_true, y_test, y_scores: precision_score(y_true, y_test, average='weighted', zero_division=0),
           'recall macro':      lambda y_true, y_test, y_scores: recall_score(y_true, y_test, average='macro', zero_division=0),
           'recall weighted':   lambda y_true, y_test, y_scores: recall_score(y_true, y_test, average='weighted', zero_division=0),
           'roc_auc_score':     lambda y_true, y_test, y_scores: roc_auc_score(y_true, y_scores, multi_class='ovr'),
           'ap_macro': lambda y_true, y_test, y_scores: average_precision_score(label_binarize(y_true,
                                                                                classes=range(len(le_t.classes_))),
                                                                                y_scores,
                                                                                average='macro'),
           'ap_micro': lambda y_true, y_test, y_scores: average_precision_score(label_binarize(y_true,
                                                                                classes=range(len(le_t.classes_))),
                                                                                y_scores,
                                                                                average='micro')
           }

trainloader, testloader, le = eutil.create_dataloaders(args.train_csv,
                                                        args.test_csv,
                                                        load_to_memory=args.load_dataset_to_memory,
                                                        batch_size=args.batch_size,
                                                        output_size=OUTPUT_SIZE,
                                                        channels=CHANNELS,
                                                        tf_array=array_transform,
                                                        tf_preload=TF_PRELOAD,
                                                        verbose=True)
# Make dir for confusion matrices
now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
os.mkdir(f'single_confmat_{now}')

dflist = []
model_base = os.path.split(args.model)[1]
Y = eutil.Results(METRICS, n_splits=1)
for hierarchy, LABEL_MAP in enumerate(LABEL_MAPS):

    # RandomForest
    print('Evaluating random forest...')
    y_tup, le_t = eutil.evaluate_model('RandomForest',
                                testloader,
                                trainloader=trainloader,
                                le=le,
                                label_map=LABEL_MAP)
    print('Done!')

    Y.add_results('RandomForest', y_tup, 0)


    for TTA in [True, False]:
        if TTA:
            n_guess = args.tta_guesses
            suffix = f'_TTA{n_guess}'
            tta_tf = TF_TTA
        else:
            suffix = ''
            n_guess = None
            tta_tf = None

        print('Loading resnet... ')
        model = eutil.get_model('ResNet',
                                    n_channels=len(CHANNELS),
                                    n_classes=N_classes,
                                    model_path=args.model,
                                    resnet_model=args.resnet_model)
        print('Done!')

        # ResNet
        print(f'Evaluating resnet{suffix}...')
        y_tup, le_t = eutil.evaluate_model('ResNet',
                                    testloader,
                                    model=model,
                                    le=le,
                                    label_map=LABEL_MAP,
                                    n_guess=n_guess,
                                    multi_tf=tta_tf)
        print("Done!")

        Y.add_results('ResNet'+suffix, y_tup, 0)

        # Average
        print(f'Evaluating average{suffix}...')
        y_tup, le_t = eutil.evaluate_model('Average',
                                    testloader,
                                    trainloader=trainloader,
                                    model=model,
                                    le=le,
                                    label_map=LABEL_MAP,
                                    n_guess=n_guess,
                                    multi_tf=tta_tf)
        print('Done!')
        Y.add_results('Average'+suffix, y_tup, 0)
    #end for

    #end for

    Y.plot_conf_matrix(list(Y.results.keys()),
                        le=le_t,
                        figsize=(25,25),
                        normalize='true',
                        savename=os.path.join(f'single_confmat_{now}',
                                      model_base+'_lvl'+str(hierarchy)))
    plt.close()

    s = Y.print_csv(list(Y.results.keys()), cross_validate=False)
    df = pd.read_csv(io.StringIO(s), sep=',')

    df.insert(1,'hierarchy',hierarchy)
    dflist.append(df)

full_df = pd.concat(dflist)
print(full_df)
now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

full_df.to_csv(f'results_{model_base}_{now}.csv', index=False)
