"""
Evaluation and result handling functions and classes
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import seaborn as sns
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torchvision import transforms

import lappieo.classificationutils as cu
import lappieo.utils as ut


def create_dataloaders(train_csv, 
                        test_csv, 
                        load_to_memory, 
                        batch_size,
                        output_size,
                        channels,
                        tf_array=None,
                        tf_preload=None,
                        tf_test=None,
                        tf_train=None,
                        verbose=True):
                        
    fnames_train, labels_train = ut.read_fname_csv(train_csv)
    fnames_test, labels_test = ut.read_fname_csv(test_csv)
    
    _, le = ut.encode_labels(labels_train+labels_test)

    N_classes = len(le.classes_)
    print(N_classes)

    y_train = le.transform(labels_train)
    y_test = le.transform(labels_test)

    trainset = cu.ImagePathDataset(fnames_train, 
                                y_train, 
                                output_size=output_size,
                                channels=channels,
                                array_transform=tf_array,
                                preload_tensor_transform=tf_preload,
                                tensor_transform=tf_train,
                                load_to_memory=load_to_memory)

    testset = cu.ImagePathDataset(fnames_test, 
                                y_test, 
                                output_size=output_size,
                                channels=channels,
                                array_transform=tf_array,
                                preload_tensor_transform=tf_preload,
                                tensor_transform=tf_test,
                                load_to_memory=load_to_memory)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    if verbose:
        print("Trainset:")
        ut.print_label_counts(y_train,le)
        print("Testset:")
        ut.print_label_counts(y_test,le)
    return trainloader, testloader, le

def get_model(name, **kwargs):
    if name == 'centerpixel':
        # Random forest center pixel extractor
        # kwargs: None
        class IdentityModel(torch.nn.Module):
            def __init__(self, transform=None):
                super().__init__()
                self.transform = transform
            def forward(self, x):
                if self.transform:
                    x = self.transform(x)
                return x.squeeze()

        model = IdentityModel(transform=transforms.CenterCrop(1))
    elif name == 'ResNet':

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ResNet model
        # kwargs:
        # n_channels
        # n_classes
        # model_path
        finetuned_resnet = cu.Sentinel2_ResNet50(N_channels=kwargs['n_channels'], 
                                                N_classes=kwargs['n_classes'],
                                                resnet_model=kwargs['resnet_model'])
        finetuned_resnet.to(device)
        finetuned_resnet.load_state_dict(torch.load(kwargs['model_path'], map_location=device))

        model = finetuned_resnet
    elif name == 'IIC':
        # IIC clustering
        # kwargs:
        # center_crop
        # n_channels
        # n_clusters
        # n_overcluster
        # model_path
        # head
        tf = transforms.Compose([
                        transforms.CenterCrop(kwargs['center_crpo'])
                        ])

        IIC_cluster = cu.ICC_Sentinel2_ResNet50(kwargs['n_channels'], kwargs['n_clusters'], kwargs['n_overcluster'], [tf, tf])
        IIC_cluster.to('cuda')
        IIC_cluster.load_state_dict(torch.load(kwargs['model_path'])) #pixwise_10epoch.pt
        IIC_cluster.eval()

        # Output head A
        class ModelAWrapper(torch.nn.Module):
            def __init__(self, orig_model):
                super().__init__()
                self.orig_model = orig_model
                
            def forward(self, x): 

                return self.orig_model(x, head='A')
        if kwargs['head'] == 'A':
            model = ModelAWrapper(IIC_cluster)
        elif kwargs['head'] =='B':
            model = IIC_cluster
        else: 
            raise Exception('Invalid head name')

    model.eval()
    return model

def get_RF_features(trainloader, testloader):
    model = get_model('centerpixel')
    y_train, _, X_train = cu.torch_predict(model, trainloader, gpu=False)
    y_test, _, X_test = cu.torch_predict(model, testloader, gpu=False)

    return X_train, y_train, X_test, y_test

def get_trained_RF(trainloader):
    clf = RandomForestClassifier()
    X_train, y_train, _, _ = get_RF_features(trainloader, trainloader)
    clf.fit(X_train, y_train)
    return clf

def rf_classify_batch(clf, batch, n_classes):
    model = get_model('centerpixel')
    output = model(batch.detach().cpu())
    X_eval = output.numpy()
    y_scores = clf.predict_proba(X_eval)

    rf_scores = np.zeros((y_scores.shape[0], n_classes)) #Trainset might not have all classes
    rf_scores[:,clf.classes_] = y_scores

    y_pred = np.argmax(rf_scores, axis=1)
    return y_pred, rf_scores

def evaluate_model(name, 
                   testloader, 
                   trainloader=None, 
                   model=None, 
                   le=None, 
                   label_map=None, 
                   n_guess=None, 
                   multi_tf=None):
    if name == 'RandomForest':
        X_train, y_train, X_test, y_test = get_RF_features(trainloader, testloader)
        clf = RandomForestClassifier()

        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_test)
        y_pred = np.argmax(y_scores, axis=1)
    
    elif name == 'ResNet':
        if not model:
            raise Exception('Model instance needed')
        y_test, y_pred, y_scores = cu.torch_predict(model, testloader, n_guess=n_guess, tf=multi_tf)
        y_scores = scipy.special.softmax(y_scores, axis=1)
            
    elif name == 'Average':
        # ResNet
        tup, _ = evaluate_model('ResNet', testloader, model=model, n_guess=n_guess, multi_tf=multi_tf)
        _, _, resnet_scores = tup
        # RF
        X_train, y_train, X_test, y_test = get_RF_features(trainloader, testloader)
        n_classes = len(np.unique(list(y_test)+list(y_train)))
        clf = RandomForestClassifier()

        clf.fit(X_train, y_train)
        y_scores_RF = clf.predict_proba(X_test)
        
        # Average
        rf_scores = np.zeros((y_scores_RF.shape[0], n_classes)) #Trainset might not have all classes
        rf_scores[:,clf.classes_] = y_scores_RF
        
        y_scores = (rf_scores + resnet_scores) /2
        y_pred = np.argmax(y_scores,axis=1)
        
    else:
        raise Exception('Invalid name')
    
    if label_map:
        y_test = le.inverse_transform(y_test)
        y_pred = le.inverse_transform(y_pred)

        y_test = list(map(label_map, y_test))
        y_pred = list(map(label_map, y_pred))
        _, le_t = ut.encode_labels(np.unique(np.vectorize(label_map)(le.classes_)))

        y_scores = ut.transform_scores(y_scores, le, le_t, label_map)
        y_test = le_t.transform(y_test)
        y_pred = le_t.transform(y_pred)
    else:
        le_t = le
    return (y_test, y_pred, y_scores), le_t


def top_k_accuracy_score(y_true, y_score, k, le=None, label_map=None):
    y_true = np.asarray(y_true)
    assert(y_true.shape[0] == y_score.shape[0])

    sorted_pred = np.argsort(y_score, axis=1, kind='heapsort')[:, ::-1]

    if le:
        hits = []
        for i in range(len(y_true)):
            c = sorted_pred[i, :k]
            c = le.inverse_transform(c) # index to label strings

            if label_map:
                c = np.vectorize(label_map)(c)
            hits.append(y_true[i] in c)
    else:
        hits = [y in sorted_pred[i, :k] for i,y in enumerate(y_true)]
    return np.average(hits)
            
def calc_metrics(y_test, y_pred, y_scores, metric_dict):
    v = {}
    for name in metric_dict.keys():
        v[name] = metric_dict[name](y_test, y_pred, y_scores)
    return v

def print_report(name, v):
    print(f"### {name}")

    for metric in v.keys():
        print(f"{metric}: {v[metric]:.3}")
    print()

def get_linestyles():
    lines = ['-', '--', '-.', ':']
    colors = ['r', 'lightblue', 'g', 'y', 'magenta', 'black', 'orange']
    linestyles = []
    for line in lines:
        for color in colors:
            linestyles.append((color,line))
    return linestyles


def macro_average(x, y, classes):
    """Calculates the mean of values in dict y
    """
    all_x = np.linspace(0,1,200)
    mean_y = np.zeros_like(all_x) #interpolated y-axis
    for i in classes:
        mean_y += np.interp(all_x, x[i], y[i]) 
    mean_y /= len(classes)

    var_y = np.zeros_like(all_x)
    for i in classes:
        var_y += np.power((np.interp(all_x, x[i], y[i]) - mean_y), 2)
    var_y = var_y/len(classes)
    std_y = np.sqrt(var_y)
    return all_x, mean_y, std_y

def calculate_curves(y_true, y_scores):
    """Calculates precision-recall curves and roc-auc curves
    """
    n_classes = y_scores.shape[1]
    precision = {}
    recall = {}
    average_precision = {}

    fpr = {}
    tpr = {}
    roc_auc = {}
    # Class-wise values
    for i in range(n_classes):
        # Values
        prec, rec, _ = precision_recall_curve(y_true[:, i],
                                            y_scores[:, i])
        precision[i] = prec[::-1] # sklearn api returns these in inverse order
        recall[i] = rec[::-1]
        
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], 
                                    y_scores[:, i])

        # Aggregates
        average_precision[i] = average_precision_score(y_true[:, i], 
                                                    y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Micro-macro averages prec-rec
    prec_micro, rec_micro, _ = precision_recall_curve(y_true.ravel(),
                                                                y_scores.ravel())
    precision['micro'] = prec_micro[::-1]
    recall['micro'] = rec_micro[::-1]
                                                                
    recall['macro'], precision['macro'], precision['macro_std'] = macro_average(recall, 
                                                            precision, 
                                                            range(n_classes))

    
    average_precision['micro'] = average_precision_score(y_true, y_scores,
                                                        average="micro")
    average_precision["macro"] = average_precision_score(y_true, y_scores,
                                                        average="macro")

    # micro-macro roc-auc
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), 
                                            y_scores.ravel())
    fpr["macro"], tpr["macro"], tpr["macro_std"] = macro_average(fpr, tpr, range(n_classes))

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    curves = {}
    curves['precision'] = precision
    curves['recall'] = recall
    curves['fpr'] = fpr
    curves['tpr'] = tpr
    curves['average_precision'] = average_precision
    curves['roc_auc'] = roc_auc

    return curves


def plot_curve(x, y, s, labels, classes, text, linestyles, classnames=None, plot_micro=True, plot_std=True):

    plt.figure(figsize=(16,12))
    plt.style.use('default')
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.plot(x['macro'], y['macro'], label=f'macro avg, {text}: {s["macro"]:.2f}', lw=3, ls='--', color='blue')

    if not classnames:
        classnames = classes
    if plot_micro:
        plt.plot(x['micro'], y['micro'], label=f'micro avg, {text}: {s["micro"]:.2f}', lw=3, ls='--', color='purple')
    if plot_std:
        prec_upper = np.minimum(y["macro"] + y["macro_std"], 1)
        prec_lower = np.maximum(y["macro"] - y["macro_std"], 0)

        plt.fill_between(x['macro'], prec_lower, prec_upper, color='grey', alpha=.2, label=r'macro avg $\pm$ 1 std')
    for i, c in enumerate(classes):
        plt.plot(x[c], 
                y[c], 
                lw=2, 
                label=classnames[i] + f', {text}: {s[c]:.2f}', 
                color=linestyles[i][0], 
                ls=linestyles[i][1],
                alpha=0.2)
    plt.xlabel(labels[0], fontsize=16)
    plt.ylabel(labels[1], fontsize=16)
    plt.legend(bbox_to_anchor=(1.04,1), prop={'size': 11})
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# def plot_and_save_curves(curves, 
#                         classes,
#                         modelname,
#                         folder):
#     linestyles = get_linestyles()

#     plot_curve(curves['fpr'], 
#                 curves['tpr'], 
#                 curves['roc_auc'], 
#                 ('fpr', 'tpr'), 
#                 classes, 
#                 linestyles)
#     plt.title(f'ROC AUC curve for {modelname}')
#     plt.savefig(f'{folder}roc_auc_{modelname}.jpg')
#     plt.close()
#     plot_curve(curves['precision'], 
#                 curves['recall'], 
#                 curves['average_precision'], 
#                 ('Recall', 'Precision'),
#                 classes, 
#                 linestyles)
#     plt.title(f'Precision-recall curve for {modelname}')
#     plt.savefig(f'{folder}pr_{modelname}.jpg')
#     plt.close()


class Results():
    def __init__(self, metrics, n_splits):
        self.metrics = metrics
        self.y = {}
        self.n_splits = n_splits
        self.results = {}
        self.curves = {}
        self.label_encoders = {}
        self.all_classes = set()

    def init_name(self, name):
        if name not in self.y.keys():
            self.y[name] = {}
            for k1 in ['true', 'pred', 'scores']:
                self.y[name][k1] = [None for _ in range(self.n_splits)]

            self.curves[name] = {}
            for ind in list(range(self.n_splits)) + ['cv']:
                self.curves[name][ind] = {}
                for k2 in ['precision', 'recall', 'fpr', 'tpr', 'average_precision', 'roc_auc']:
                    self.curves[name][ind][k2] = {}

            self.results[name] = {}
            for metric in self.metrics.keys():
                self.results[name][metric] = [None for _ in range(self.n_splits)]

    def add_results(self, name, tup, ind=0):
        self.init_name(name)

        self.y[name]["true"][ind] = tup[0]
        self.y[name]["pred"][ind] = tup[1]
        self.y[name]["scores"][ind] = tup[2]

        # Calc
        v = calc_metrics(tup[0], tup[1], tup[2], self.metrics)
        for metric in self.metrics.keys():
            self.results[name][metric][ind] = v[metric]

    def get_splits_with_class(self , name, cls):
        splits = []
        for i in range(self.n_splits):
            if cls in self.curves[name][i]['precision'].keys():
                splits.append(i)
        return splits

    def calculate_curves(self, name, y_tup, ind, le):

        d = calculate_curves(label_binarize(y_tup[0],
                                            classes=range(y_tup[2].shape[1])), 
                                            y_tup[2])
        for k in d.keys():
            for ci in d[k].keys():
                if isinstance(ci, int):
                    c = le.inverse_transform([ci])[0]
                    if c not in self.curves[name][ind][k].keys():
                        self.curves[name][ind][k][c] = {}
                        self.all_classes.add(c)
                    self.curves[name][ind][k][c] = d[k][ci]
                else:
                    if ci not in self.curves[name][ind][k].keys():
                        self.curves[name][ind][k][ci] = {}
                    self.curves[name][ind][k][ci] = d[k][ci]
        self.label_encoders[ind] = le

    def cross_validate_curves(self, name, cls, splits, typ):
        if typ == 'prec_rec':
            x = 'recall'
            y = 'precision'
            s = 'average_precision'
        elif typ == 'roc_auc':
            x = 'fpr'
            y = 'tpr'
            s = 'roc_auc'
        else:
            raise Exception("Invalid type")

        x_cv = {}
        y_cv = {}
        s_cv = {}

        for i in splits:
            x_cv[i] = self.curves[name][i][x][cls]
            y_cv[i] = self.curves[name][i][y][cls]
            s_cv[i] = self.curves[name][i][s][cls]
            
        x_cv['macro'], y_cv['macro'], y_cv['macro_std'] = macro_average(x_cv, 
                                                                y_cv, 
                                                                splits)
        s_cv['macro'] = np.mean(np.asarray([x for x in s_cv.values()], dtype=object))
        return x_cv, y_cv, s_cv

    def cross_validate_all_curves(self, names):
        for name in names:
            d = self.curves[name]['cv']
            for typ in ['prec_rec', 'roc_auc']:
                if typ =='prec_rec':
                    x = 'recall'
                    y = 'precision'
                    s = 'average_precision'
                elif typ =='roc_auc':
                    x = 'fpr'
                    y = 'tpr'
                    s = 'roc_auc'
                
                for cls in list(self.all_classes):
                    splits = self.get_splits_with_class(name, cls)
                    x_cv, y_cv, s_cv = self.cross_validate_curves(name, cls, splits, typ)
                    d[x][cls] = x_cv['macro']
                    d[y][cls] = y_cv['macro']
                    d[s][cls] = s_cv['macro']

                splits = self.get_splits_with_class(name, 'macro')
                x_cv, y_cv, s_cv = self.cross_validate_curves(name, 'macro', splits, typ)
                d[x]['macro'] = x_cv['macro']
                d[y]['macro'] = y_cv['macro']
                d[s]['macro'] = s_cv['macro']

                splits = self.get_splits_with_class(name, 'micro')
                x_cv, y_cv, s_cv = self.cross_validate_curves(name, 'micro', splits, typ)
                d[x]['micro'] = x_cv['macro']
                d[y]['micro'] = y_cv['macro']
                d[s]['micro'] = s_cv['macro']

    def plot_curve(self, name, ind, typ, title=None):
        if typ =='prec_rec':
            labels = ('Recall', 'Precision')
            text = 'AP'
            x = 'recall'
            y = 'precision'
            s = 'average_precision'
        elif typ =='roc_auc':
            labels = ('FPR', 'TPR')
            text = 'AUC'
            x = 'fpr'
            y = 'tpr'
            s = 'roc_auc'

        if ind == 'cv':
            classes = list(self.all_classes)
            std = False
        else:
            classes = self.label_encoders[ind].classes_
            std = True
        if not title:
            title= name
        d = self.curves[name][ind]
        plot_curve(d[x], d[y], d[s], 
                                labels,
                                classes,
                                text, 
                                get_linestyles(),
                                plot_std=std)
        plt.title(title, fontsize=16)


    def plot_cross_validated_curve(self, name, cls, typ):
        if typ =='prec_rec':
            labels = ('Recall', 'Precision')
            text = 'AP'
        elif typ =='roc_auc':
            labels = ('FPR', 'TPR')
            text = 'AUC'

        splits = self.get_splits_with_class(name, cls)
        x_cv, y_cv, s_cv = self.cross_validate_curves(name, cls, splits, typ)
        plot_curve(x_cv, y_cv, s_cv, labels,
                                splits, 
                                text, 
                                get_linestyles(), 
                                classnames=["split "+str(x) for x  in splits], 
                                plot_micro=False)
        plt.title(cls)

    def plot_comparison_curve(self, names, cls, typ, title=None):
        if typ =='prec_rec':
            labels = ('Recall', 'Precision')
            text = 'AP'
        elif typ =='roc_auc':
            labels = ('FPR', 'TPR')
            text = 'AUC'
        if not title:
            title = cls

        fig, ax = plt.subplots(figsize=(12,12))
        plt.style.use('default')
        plt.axis([-0.05, 1.05, -0.05, 1.05])
        for name in names:
            splits = self.get_splits_with_class(name, cls)
            x_cv, y_cv, s_cv = self.cross_validate_curves(name, cls, splits, typ)
            ax.plot(x_cv['macro'], y_cv['macro'], label=f'{name}, {text}: {s_cv["macro"]:.2f}', lw=2)
        plt.xlabel(labels[0], fontsize=16)
        plt.ylabel(labels[1], fontsize=16)
        plt.title(title, fontsize=16)
        plt.legend(prop={'size': 10})

    def plot_and_save_curves(self, title, savename):
        for m in ['micro', 'macro']:
            for typ in ['prec_rec', 'roc_auc']:
                self.plot_comparison_curve(self.curves.keys(), m, typ, title=f'{title} {typ} {m} curve')
                plt.savefig(f'{savename}_comparison_{typ}_{m}.jpg')
                plt.savefig(f'{savename}_comparison_{typ}_{m}.pdf')
                plt.close()
        for name in self.curves.keys():
            self.plot_curve(name, 'cv', 'prec_rec', title=f'{title} precision recall-curve')
            plt.savefig(f'{savename}_prec_rec_{name}_cv.jpg')
            plt.savefig(f'{savename}_prec_rec_{name}_cv.pdf')
            plt.close()
            self.plot_curve(name, 'cv', 'roc_auc', title=f'{title} ROC AUC -curve')
            plt.savefig(f'{savename}_roc_auc_{name}_cv.jpg')
            plt.savefig(f'{savename}_roc_auc_{name}_cv.pdf')
            plt.close()


    def add_single_result(self, name, metric, ind, value):
        self.init_name(name)
        self.results[name][metric][ind] = value

    def get_y(self, name, y_type, ind):
        return self.y[name][y_type][ind]

    def print_results(self, name, ind):
        print(f"\n\n{name}, split: {ind}\n")
        for metric in self.results[name].keys():
            print(f"{metric}: {self.results[name][metric][ind]:.3}")
        print()
        
    def print_csv(self, models, cross_validate=True):
        s = "name,split,metric,val\n"
        for name in models:
            for ind in range(self.n_splits):
                for metric in self.results[name].keys():
                    s = s+f"{name},{ind},{metric},{self.results[name][metric][ind]:.3}\n"
            
            if cross_validate:
                for metric in self.results[name].keys():
                    val = np.mean(self.results[name][metric])
                    s = s+f"{name}_cv,-1,{metric},{val:.3}\n"
        return s

    def print_crossval_results(self, name):
        print(f"\n\n{name} over {self.n_splits} folds:\n")
        for metric in self.results[name].keys():
            val = np.mean(self.results[name][metric])
            print(f"{metric}: {val:.3}")
            
    def plot_conf_matrix(self, models, le=None, figsize=(10,10),normalize=None, savename=None):
        labels = le.classes_
        for name in models:
            cm = np.zeros((len(labels), len(labels), self.n_splits))
            for split in range(self.n_splits):
                cm[:,:,split] = confusion_matrix(
                                        le.inverse_transform(self.y[name]['true'][split]), 
                                        le.inverse_transform(self.y[name]['pred'][split]),
                                        labels = labels,
                                        normalize=normalize)
                
            plt.figure(figsize=figsize)
            ax = plt.subplot()
            sns.set(font_scale=(0.6+(1.0/len(labels))))
            sns.heatmap(cm.mean(axis=2), 
                        annot=True,
                        ax=ax,
                        cmap='YlGnBu',
                        fmt=".2f",
                        cbar=False)
            ax.xaxis.set_ticklabels(labels)
            ax.yaxis.set_ticklabels(labels)
            _ = plt.yticks(rotation=0, fontsize=10)
            _ = plt.xticks(rotation=90, fontsize=10)
            _ = plt.xlabel('Predicted label', fontsize=10)
            _ = plt.ylabel('True label', fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if savename:
                plt.title(f'{savename}_{name}')
                plt.savefig(f'{savename}_{name}.jpg')
                plt.savefig(f'{savename}_{name}.pdf')
                plt.close()
