""" Evaluates multiple models
"""
import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize
from torchvision import transforms

import dataset_stats
import src.evalutils as eutil
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

N_splits = 5
N_GUESS = 5

CHANNELS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
OUTPUT_SIZE=(49,49)
CENTER_CROP = 19

STATS0 = dataset_stats.stats['v3']
STATS = {}
STATS['mean'] = STATS0['mean'][CHANNELS]
STATS['std'] = STATS0['std'][CHANNELS]

def remove_nodata(img):
    img[img==STATS0['nodata']] = 0
    return img

array_transform = remove_nodata


TF_PRELOAD = transforms.Compose([
                    transforms.Normalize(STATS['mean'], STATS['std']),
                    transforms.CenterCrop(CENTER_CROP)
                    ])

TF_NOCROP = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()
                    ])

cropmax = (CENTER_CROP+1)//2
crop_scales = np.ceil(np.linspace(0,cropmax-1,5))*2+1
print("crop scales", crop_scales)

TF_CROP = []
for ti in range(N_GUESS):
    TF_CROP.append(transforms.Compose([
        TF_NOCROP,
        transforms.CenterCrop(crop_scales[ti])
    ]))


label = 'nat'
#%%
# MODELS = [{'model': lambda split: f'batchjobs/v3_100m_distill_crop_{label}{split}_00.pt',
#             'data' :lambda split: f'batchjobs/100m_{label}{split}',
#             'arch': 'resnet18',
#             'tta_tf': TF_CROP}]
 

MODELS = [{'model': lambda split: f'batchjobs/v3_100m_base_{label}{split}.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_NOCROP},

          {'model': lambda split: f'batchjobs/v3_100m_base_crop_{label}{split}.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_CROP},

          {'model': lambda split: f'batchjobs/v3_100m_trans_{label}{split}.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_NOCROP},

          {'model': lambda split: f'batchjobs/v3_100m_trans_nofrz_{label}{split}.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_NOCROP},

          {'model': lambda split: f'batchjobs/v3_100m_trans_crop_{label}{split}.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_CROP},

          {'model': lambda split: f'batchjobs/v3_100m_trans_crop_nofrz_{label}{split}.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_CROP},

          {'model': lambda split: f'batchjobs/v3_100m_distill_{label}{split}_00.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_NOCROP},

          {'model': lambda split: f'batchjobs/v3_100m_distill_nofrz_{label}{split}_00.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_NOCROP},

          {'model': lambda split: f'batchjobs/v3_100m_distill_crop_{label}{split}_00.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_CROP},

          {'model': lambda split: f'batchjobs/v3_100m_distill_crop_nofrz_{label}{split}_00.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_CROP},

          {'model': lambda split: f'batchjobs/v3_100m_iic_{label}{split}.pt',
            'data' :lambda split: f'batchjobs/100m_{label}{split}',
            'arch': 'resnet18',
            'tta_tf': TF_NOCROP}
         ]


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
           'roc_auc_macro':     lambda y_true, y_test, y_scores: roc_auc_score(y_true, y_scores, multi_class='ovr'),
           'roc_auc_micro':     lambda y_true, y_test, y_scores: roc_auc_score(label_binarize(y_true,
                                                                                classes=range(len(le_t.classes_))).ravel(), 
                                                                                y_scores.ravel()),
           'ap_macro': lambda y_true, y_test, y_scores: average_precision_score(label_binarize(y_true,
                                                                                classes=range(len(le_t.classes_))),
                                                                                y_scores,
                                                                                average='macro'),
           'ap_micro': lambda y_true, y_test, y_scores: average_precision_score(label_binarize(y_true,
                                                                                classes=range(len(le_t.classes_))),
                                                                                y_scores,
                                                                                average='micro')
           }

print("Combinations:", len(MODELS)*len(LABEL_MAPS)*(N_splits+1)*len(METRICS)*1*1) #3 Models, TTA True/False


# Check models
for d in MODELS:
    for split in range(N_splits):
        assert os.path.exists(d['model'](split)),f"{d['model'](split)} does not exist!"
        print(d['model'](split), "ok!")

# Make dir for confusion matrices
now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
matfolder = f'confmat_{now}'
os.mkdir(matfolder)
curvefolder = f'curves_{now}'
os.mkdir(curvefolder)
#%%
# Evaluate
dflist = []
prev_data = ''
for d in MODELS:
    basename = os.path.splitext(os.path.split(d['model'](''))[1])[0]
    if d['data']('') != prev_data:
        trainloaders = {}
        testloaders = {}
        label_encoders = {}
        transformed_label_encoders = {}
        n_classes = {}

        for split in range(N_splits):
            dataset = d['data'](split)
            trainloader, testloader, le = eutil.create_dataloaders(dataset+'_train.csv',
                                                                    dataset+'_test.csv',
                                                                    load_to_memory=True,
                                                                    batch_size=183,
                                                                    output_size=OUTPUT_SIZE,
                                                                    channels=CHANNELS,
                                                                    tf_array=array_transform,
                                                                    tf_preload=TF_PRELOAD,
                                                                    verbose=False)
            trainloaders[split] = trainloader
            testloaders[split] = testloader
            label_encoders[split] = le
            n_classes[split] = len(le.classes_)
        #end for
    #end if

    for hierarchy, LABEL_MAP in enumerate(LABEL_MAPS):
        Y = eutil.Results(METRICS, n_splits=N_splits)
        for split in range(N_splits):
            testloader = testloaders[split]
            trainloader = trainloaders[split]
            le = label_encoders[split]
            # RandomForest
            print('eval random forest...')
            y_tup, le_t = eutil.evaluate_model('RandomForest',
                                        testloader,
                                        trainloader=trainloader,
                                        le=le,
                                        label_map=LABEL_MAP)
            transformed_label_encoders[split] = le_t
            Y.add_results('RandomForest', y_tup, split)
            Y.calculate_curves('RandomForest', y_tup, split, le_t)

            for TTA in [True, False]:
                if TTA:
                    suffix = f'_TTA{N_GUESS}'
                    n_guess = N_GUESS
                    tta_tf = d['tta_tf']
                else:
                    suffix = ''
                    n_guess = None
                    tta_tf = None

                print('load resnet...')
                model = eutil.get_model('ResNet',
                                            n_channels=len(CHANNELS),
                                            n_classes=n_classes[split],
                                            model_path=d['model'](split),
                                            resnet_model=d['arch'])

                # ResNet
                print('eval resnet...')
                y_tup, le_t = eutil.evaluate_model('ResNet',
                                            testloader,
                                            model=model,
                                            le=le,
                                            label_map=LABEL_MAP,
                                            n_guess=n_guess,
                                            multi_tf=tta_tf)
                Y.add_results('ResNet'+suffix, y_tup, split)
                Y.calculate_curves('ResNet'+suffix, y_tup, split, le_t)


                # Average
                print('eval average...')
                y_tup, le_t = eutil.evaluate_model('Average',
                                            testloader,
                                            trainloader=trainloader,
                                            model=model,
                                            le=le,
                                            label_map=LABEL_MAP,
                                            n_guess=n_guess,
                                            multi_tf=tta_tf)
                Y.add_results('Average'+suffix, y_tup, split)
                Y.calculate_curves('Average'+suffix, y_tup, split, le_t)
            #end for
        #end for
        Y.cross_validate_all_curves(Y.curves.keys())
        curvename = basename+'_lvl'+str(hierarchy)
        Y.plot_and_save_curves(curvename, f'{curvefolder}/curve_{curvename}')

        print(d['data'](''), basename, f'lvl {hierarchy}')
        for norm in ['true', None]:
            matname = basename+'_lvl'+str(hierarchy)+'_norm-'+str(norm)
            Y.plot_conf_matrix(list(Y.results.keys()),
                               le=le_t,
                               figsize=(10,10),
                               normalize=norm,
                               savename=os.path.join(matfolder, matname))
            plt.close()

        s = Y.print_csv(list(Y.results.keys()))
        df = pd.read_csv(io.StringIO(s), sep=',')

        df.insert(1,'hierarchy',hierarchy)
        df.insert(0,'data',d['data'](''))
        df.insert(0,'model',basename)
        dflist.append(df)
    #end for
    prev_data = d['data']('')

full_df = pd.concat(dflist)

full_df.to_csv(f'results_{label}_{now}.csv', index=False, sep=';', decimal=',')
