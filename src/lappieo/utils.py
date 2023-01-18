# -*- coding: utf-8 -*-
"""
Utility functions and classes for geotiff handling and processing,
"""

import os

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import torch
from rasterio.mask import mask
from shapely.geometry import mapping
from skimage.color import label2rgb
from skimage.exposure import equalize_hist, rescale_intensity
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

##################### READING / WRITING #####################


def show_gpkg_layers(fname):
    for layername in fiona.listlayers(fname):
        with fiona.open(fname, layer=layername) as src:
            print(layername, len(src))

def read_raster_for_classification(fname):

    src = rasterio.open(fname)
    array = src.read()
    array = np.moveaxis(array,0,2)
    return array, src

def write_classified_raster(fname, src, C):
    out_img = np.expand_dims(C,0).astype(np.uint16)

    out_meta = src.meta
    out_meta.update({'count': 1, 
                     'dtype': 'uint16',
                     'nodata': 65535})

    with rasterio.open(fname + '.tif', "w", **out_meta) as dest:
        dest.write(out_img)

    rgb = label2rgb(C)
    plt.imsave(fname + '.png', rgb)

def write_float_raster(fname, src, A, channels):
    out_img = A.astype(np.uint8)

    out_meta = src.meta
    out_meta.update({'count': channels, 
                     'dtype': 'uint8',
                     'nodata': 255})

    with rasterio.open(fname + '.tif', "w", **out_meta) as dest:
        dest.write(out_img)

def read_sampled_data(gdf, data_col, label_col):
    gdf.loc[:,label_col] = gdf[label_col].replace(0, np.nan)
    gdf = gdf.dropna(subset=[label_col])


    if np.issubdtype(gdf[label_col].dtype, np.float):
        gdf.loc[:,label_col] = gdf.loc[:,label_col].astype(int).apply(str)


    X = gdf.iloc[:,data_col].fillna(0).to_numpy()

    y = gdf[label_col].tolist()

    return X, y

def read_tif_filenames(df, fid_col, label_col, tif_folder):
    """
    From a reference DataFrame, reads file paths and their corresponding labels
    from a specified folder, checking that they exist.

    Parameters
    ----------
    df : pd.DataFrame
        reference dataframe.
    fid_col : string
        name of the column containing filename information.
    label_col : string
        name of column containing label information.
    tif_folder : string
        location of the tif-files corresponding to fid_col.

    Raises
    ------
    ValueError
        If a file from fid_col is not found from tif_folder, raises error.

    Returns
    -------
    fnames : list
        list of filenames corresponding to fid_col.
    labels : list
        list of labels corresponding to label_col.

    """

    fnames = []
    labels = []
    for i in range(len(df)):
        row = df.iloc[i,:]
        fid = str(row[fid_col])

        fname = os.path.abspath(os.path.join(tif_folder, fid+'.tif'))
        if not os.path.exists(fname):
            raise ValueError('File {} does not exist'.format(fname))
        label = str(row[label_col])

        fnames.append(fname)
        labels.append(label)

    return fnames, labels

def interpret_natura_code(source, reference_loc='natura_koodit.csv'):

    df = pd.read_csv(reference_loc, delimiter=',')
    if source.isdigit() and source in df['Koodi_txt'].values:
        target = df[df['Koodi_txt']==source]['Nimi'].values[0]
    elif source in df['Nimi'].values:
        target = df[df['Nimi']==source]['Koodi_txt'].values[0]
    else:
        target = ''
    return target

def recover_raster_from_tensor(T, mean, std):
    A = np.moveaxis(T.numpy(),0,2)
    A = A*std + mean
    return A

def read_fname_csv(csv_fname):
    df = pd.read_csv(csv_fname)

    fnames = df['fname'].map(str).tolist()
    labels = df['label'].map(str).tolist()

    return fnames, labels
##################### PLOTTING / PRINTING #####################

def print_label_counts(y, le):
    d = {'count': np.histogram(y, bins=len(le.classes_))[0],
         'label': le.classes_,
         'y value': le.transform(le.classes_)
    }

    df = pd.DataFrame.from_dict(d)
    print(df)

def plot_conf_matrix(y_true, y_pred, le=None, labels=None, figsize=(10,8), cmap='YlGnBu', normalize=None, cm=None):
    """
    Plots a confusion matrix with inverse label transform using a label
    encoder le
    """
    # Check if vectors are in label or index form
    try:
        _ = y_true.astype(float)
        y_true = le.inverse_transform(y_true)
        y_pred = le.inverse_transform(y_pred)
    except:
        pass

    if normalize:
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    ax = plt.subplot()
    cm = confusion_matrix(y_true, 
                        y_pred, 
                        labels=labels,
                        normalize=normalize)
    sns.heatmap(cm, 
                annot=True, 
                ax=ax, 
                cmap=cmap,
                fmt=fmt,
                cbar=False)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    _ = plt.yticks(rotation=0)
    _ = plt.xticks(rotation=90)
    _ = plt.xlabel('Predicted label')
    _ = plt.ylabel('True label')

def show(I, equalize=True):
    """
    Shows arbitary numpy array with possible equalization.
    Useful for arrays with > 3 channels i.e. multispectral imagery.

    Parameters
    ----------
    I : np.ndarray
        Numpy array.
    equalize : Bool, optional
        Whether to equalize the input. The default is True.

    Returns
    -------
    None.

    """
    I = rescale_intensity(I[:,:,:3], out_range=np.uint8)

    if equalize:
        I = np.uint8(equalize_hist(I)*255)
    plt.imshow(I)

def show_batch_montage(x_batch, STATS):
    for i in range(3*3):
        x = x_batch[i,::]
        plt.subplot(3,3,i+1)
        img = recover_raster_from_tensor(x, STATS['mean'], STATS['std'])
        show(img[:,:,[3,2,1]], equalize=True)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

##################### DATA MANIPULATION #####################
def preprocess_gdf(gdf, label_col, func=int):
    """
    Removes 0 and nan values from the selected column and applies
    a function (default: int(x)) to the column

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame to be processed.
    label_col : string
        Columnto be processed.

    Returns
    -------
    gdf : GeoDataFrame
        Processed geodataframe.

    """
    gdf.loc[:,label_col] = gdf[label_col].replace(0, np.nan)
    gdf = gdf.dropna(subset=[label_col])

    gdf[label_col] = gdf[label_col].map(lambda x: func(x))
    return gdf

def prune_dataset_by_label(X, y, y_remove):
    wrong_inds = np.where([(x in y_remove) for x in y])[0]

    X = np.delete(X,wrong_inds, axis=0)
    y = np.delete(y,wrong_inds)

    return  X, y

def dataset_statistics(fnames, nodata_val=2147483647):
    maxval = 0
    minval = np.inf

    meanlist = []
    stdlist = []

    for fname in tqdm(fnames):
        img, src = read_raster_for_classification(fname)
        img[img==nodata_val] = 0
        mean = np.mean(np.mean(img, axis=0),axis=0)
        std = np.std(np.std(img, axis=0),axis=0)

        meanlist.append(mean)
        stdlist.append(std)

        max_ = np.max(img)
        min_ = np.min(img)
        if max_ > maxval:
            maxval = max_
        if min_ < minval:
            minval = min_

    meanlist = np.asarray(meanlist)
    stdlist = np.asarray(stdlist)

    fullmean = meanlist.mean(axis=0)
    fullstd = np.sqrt(np.mean(stdlist**2,axis=0))

    d = {'nodata': nodata_val,
         'mean': fullmean,
         'std': fullstd}
    return d

def encode_labels(s):
    """
    Encodes the N unique labels of a pandas Series to values in range
    0 to N-1

    Parameters
    ----------
    s : list of labels

    Returns
    -------
    y : np.ndarray
        1d vector with labels from 0 to N-1.
    le : LabelEncoder
        LabelEncoder object.

    """

    le = preprocessing.LabelEncoder()
    le.fit(s)
    y = le.transform(s)

    return y, le

def transform_scores(scores, le, le_t, label_map):
    t = {}
    # Go through mapped classes and initalize list
    for ct in le_t.classes_:
        t[ct] = []
		
    # Go through original classes and save them to the list corresponding 
    # the mapped label
    for c in le.classes_:
        i = le.transform([c])[0]
        ct = label_map(c)
        t[ct].append(i)

    new_scores = np.zeros((scores.shape[0], len(t.keys())))
    for ct in le_t.classes_:
        i_t = le_t.transform([ct])[0]
        new_scores[:,i_t] = scores[:,t[ct]].sum(axis=1)
    return new_scores

def csv_train_test_split(csv_fname, split_n, **kwargs):
    fnames, labels = read_fname_csv(csv_fname)

    skf = StratifiedKFold(**kwargs)

    index_list = [idx for idx in skf.split(fnames,labels)]

    train_idx, test_idx = index_list[split_n]

    fnames_train = [fnames[i] for i in train_idx]
    labels_train = [labels[i] for i in train_idx]
    fnames_test = [fnames[i] for i in test_idx]
    labels_test = [labels[i] for i in test_idx]

    train_df = pd.DataFrame(list(zip(fnames_train, labels_train)), columns=['fname', 'label'])
    test_df =  pd.DataFrame(list(zip(fnames_test, labels_test)), columns=['fname', 'label'])

    print("len train: ", len(train_df), "\nlen test: ", len(test_df), "\np: ", len(test_df)/len(fnames))

    temp = os.path.splitext(csv_fname)

    trainname = f"{temp[0]}{split_n}_train{temp[1]}"
    testname = f"{temp[0]}{split_n}_test{temp[1]}"

    train_df.to_csv(trainname, index=False)
    test_df.to_csv(testname, index=False)
    print(trainname, testname)


# Transforms

# Optional gaussian vignetting
def gaussian_kernel(size, sigma, normalize=False):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    if normalize:
        g = g/g.sum()
    return g

def gaussian_vignette(X, size, sigma):
    G = gaussian_kernel(size, sigma)
    return (X*G).type(torch.float)


def img_to_batch(sample, k=3, gpu=True):

    pad = k//2
    C,H,W = sample.shape
    batch = torch.zeros((H - 2*pad) * (W - 2*pad), C, k, k)
    if gpu:
        batch = batch.to('cuda')
    ind = 0
    for i in range(pad, H - pad):
        for j in range(pad, W - pad):
            b = sample[:, i-pad:i+pad+1, j-pad:j+pad+1]
            batch[ind, ::] = b
            ind += 1
    return batch

def out_to_img(out):
    H = int(np.sqrt(out.shape[0]))
    out = torch.reshape(out, (H,H, out.shape[1]))
    out = out.permute(2,0,1)
    return out

def segment_img(img, k, net):
    assert img.shape[1] == img.shape[2]

    batch = img_to_batch(img, k)

    with torch.no_grad():
        out = net(batch)

    outimg = out_to_img(out)
    _, S = torch.max(outimg, dim=0)

    return outimg, S.detach().cpu().numpy()

#S2_S1_KORK_MEAN = torch.tensor([ -8.1832,  -5.6993, -10.6945, -14.7571, -13.9787, -14.1193, -13.7243,
#        -15.5303, -14.6463,  16.2979,  20.8292,  27.0699,  -4.2888])

def segment_large_tensor(T, model, classes, k=9, block_size=500):
    """Yes this is horrible but it works
    """
    def _segblock(si,ei,sj,ej, segmented, outputimg):
        """Helper function
        """
        img = T[:, si:ei , sj:ej ]
        #chann_mean = torch.mean(torch.mean(img,dim=1),dim=1)
	#or (torch.sum(torch.abs(chann_mean - S2_S1_KORK_MEAN)) <= 0.003)
        if torch.all(img==0): 
            print("Skip null square")
            S_img = np.zeros((Ssz, Ssz))
            O_img = torch.zeros(classes, Ssz, Ssz)
        else:
            O_img, S_img = segment_img(img, k, model)
        segmented[si + pad: ei - pad, sj + pad : ej - pad] = S_img
        outputimg[:, si + pad: ei - pad, sj + pad : ej - pad] = O_img
        return segmented, outputimg

    C,H,W = T.shape
    pad = k//2
    Ssz = block_size - 2*pad

    segmented = np.zeros((H, W))
    outputimg = torch.zeros(classes, H, W)

    for i in tqdm(np.arange(0,H//2,Ssz)):
        for j in tqdm(np.arange(0,W//2,Ssz)):
            # Process block and its mirror counterpart to minimize border pad
            si = int(i)
            ei = int(si+block_size)
            sj = int(j)
            ej = int(sj+block_size)

            segmented, outputimg = _segblock(si,ei,sj,ej, segmented, outputimg)

            # Mirror
            ei = int(H-i)
            si = int(ei-block_size)
            ej = int(W-j)
            sj = int(ej-block_size)

            segmented, outputimg = _segblock(si,ei,sj,ej, segmented, outputimg)

            si = int(i)
            ei = int(si+block_size)
            ej = int(W-j)
            sj = int(ej-block_size)

            segmented, outputimg = _segblock(si,ei,sj,ej, segmented, outputimg)

            # Mirror
            ei = int(H-i)
            si = int(ei-block_size)
            sj = int(j)
            ej = int(sj+block_size)

            segmented, outputimg = _segblock(si,ei,sj,ej, segmented, outputimg)

    return segmented, outputimg


def Convert(tup):
    """Converts list of tuples to a dictionary
    """
    di = {}
    for a, b in tup:
        di.setdefault(a, b)
    return di

def make_cluster_map(y_orig, y_pred, visualize=False):
    if visualize:
        print("Original unique values:")
        print(np.unique(y_orig))
        print(np.unique(y_pred))

    #Label encoding
    y_dummy, le_dummy = encode_labels(y_orig)
    y_pred_dummy, le_pred_dummy = encode_labels(y_pred)

    inv_orig = lambda x: le_dummy.inverse_transform(x)
    inv_pred = lambda x: le_pred_dummy.inverse_transform(x)

    contmat = contingency_matrix(y_dummy, y_pred_dummy)
    if visualize:
        sns.heatmap(contmat, annot=True, cbar=False)

    y_map_dummy = contmat.argmax(0)

    y_map = [
            (inv_pred( [i] )[0], inv_orig( [y_map_dummy[i]] )[0])
            for i in np.unique(y_pred_dummy)
            ]

    return Convert(y_map)

##################### RASTER MANIPULATION #####################

def remove_border(A):
    xs,ys,zs = np.where(A!=0)
    return A[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1]

def clip_raster_and_save(src, mask_gdf, id_col, save_output=False, output_folder='output'):
    """
    Go through each shape, crop it from the raster and save output to file and
    to the dataframe
    """

    mask_gdf = mask_gdf.copy()
    tiff_name_list = []
    tiff_dir = {}

    if save_output and not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i in tqdm(range(len(mask_gdf))):
        geom = mask_gdf.geometry[i]
        id_= mask_gdf[id_col][i]

        geom_mapping = [mapping(geom)]

        # Extract raster
        out_image, out_transform = mask(src, geom_mapping, crop=True)
        out_image = remove_border(out_image)

        # Save file information to dataframe
        out_fname = "{:06d}.tif".format(id_)
        tiff_name_list.append(out_fname)

        if id_ in tiff_dir.keys():
            raise ValueError('Id column is not unique')
        tiff_dir[id_] = out_image

        # Save output
        if save_output:
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})


            with rasterio.open(os.path.join(output_folder,out_fname), "w", **out_meta) as dest:
                dest.write(out_image)

    mask_gdf['tiff'] = tiff_name_list

    return mask_gdf, tiff_dir


def sample_raster(src, gdf, label_col):
    """
    Samples a geotiff raster according to a shapefile containing point
    information

    Parameters
    ----------
    src : source geotiff raster

    gdf : shapefile geodataframe containing point geometry information

    label_col : pandas column name with labels

    Returns
    -------
    X : NxM np.ndarray where N is amount of points in gdf and M amount of
    channels in src

    y : 1d vector containing encoded label values from series

    """

    gdf = gdf.dropna(subset=[label_col])
    # Label encoding
    y, le = encode_labels(gdf[label_col])

    # Initialize X
    X = np.zeros((len(gdf), src.count))

    # Go through each point in gdf and sample corresponding raster value
    for i in tqdm(range(len(gdf))):
        geom = gdf.geometry.iloc[i]

        geom_mapping = [mapping(geom)]

        values, _ = mask(src, geom_mapping, crop=True)

        X[i,:] = values.squeeze()

    return X, y
