from skimage.color import label2rgb
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import rioxarray
import rasterio as rio
from dask.diagnostics import ProgressBar
from dask.distributed import progress, wait
import dask
import dask.array as da
from pprint import pprint
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def inference(a: dask.array, clf) -> dask.array:
    c0 = clf.predict_proba(a)
    c = (254*c0).astype(np.uint8)
    return c

def read_masked_data(A: dask.array):
    """ Picks nonzero values along depth from A and returns rows of nonzero values
    and their index mask
    
    Performance depends highly on the mask rechunking size. If the chunk size is too large, memory 
    use per worker is too high, and if it is too low, chunks are probably copied across workers so
    that system memory usage is too high and SLURM job crashes.
    
    params
    A : array
    
    returns
    data: nonzero rows
    mask: row locations
    """

    A0 = da.moveaxis(A,0,2)
    ny, nx, chan = A0.shape
    a = A0.reshape(ny*nx, chan)

    mask = da.where(~da.all(a==0,axis=1))[0]
    mask.compute_chunk_sizes()
    mask = mask.rechunk((1e6,-1)) #inference float chunks get too big otherwise
    
    data = a[mask,:]
    return data, mask

def masked_inference(A: dask.array, clf)->dask.array:
    """Classifies an array depth-wise 
    """
    
    # Masking
    data, mask = read_masked_data(A)
    
    # New data
    chan, ny, nx = A.shape
    c = da.zeros((ny*nx, len(clf.classes_)), dtype=np.uint8) #empty array for results
    
    # Inference
    if len(data)!=0:
        c0 = inference(data, clf)
        c[mask,:] = c0

    # Inverse reshape
    C = c.reshape(ny,nx, -1)
    C = da.moveaxis(C,2,0)
    
    return C

if __name__=='__main__':
    from dask.distributed import Client, LocalCluster, Lock
    #n_workers
    threads_per_worker = 1
    #memory_limit
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(processes=True, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    client.cluster

    fname = 'B_test.tif'
    
    chunk_s = 2**11
    xds = rioxarray.open_rasterio(fname, 
                                  chunks={'band': -1, 'x': chunk_s, 'y': chunk_s},
                                  lock=False,
                                  parallel=True)
    
    df = pd.read_csv('../../processed/traindata_koealat_20221_at_221221_b10m_25jan_acl.dbf.csv')

    
    dfY = df.iloc[:,0]
    dfX = df.iloc[:,1:]

    # Print info
    print("Columns. First one is chosen as target")
    print("Index\t\tColumn")
    for i, col in enumerate(df.columns):
        print(f"{i}\t\t{col}")
    print()

    print("\nTarget class distribution")
    print("label\tcount")
    print(dfY.value_counts())
    print()

    # Classes smaller than 6 are removed
    drop_classes = dfY.value_counts()[dfY.value_counts()<6].index.values
    drop_series = ~dfY.isin(drop_classes)

    dfY = dfY.loc[drop_series]
    dfX = dfX.loc[drop_series,:]

    print("Classes smaller than 6 are removed:")
    print(drop_classes)
    print()

    # Final dataset
    X = dfX.to_numpy(dtype=xds.dtype)
    y = dfY.to_numpy(dtype=xds.dtype)

    le = LabelEncoder()
    y = le.fit_transform(y)

    print(f"Shape of X: {X.shape}")
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print('label mapping:')
    pprint(le_name_mapping)
    
    from dask_ml.wrappers import ParallelPostFit

    #clf_rf = ParallelPostFit(estimator=rf_random.best_estimator_)
    clf = ParallelPostFit(estimator=RandomForestClassifier(n_jobs=1))
    clf.fit(X, y)
    
    B = da.asarray(xds)
    
    classes = le.inverse_transform(clf.classes_)
    
    t = datetime.now()
    C = masked_inference(B, clf).persist()
    progress(C)
    print(datetime.now()-t)

    def da_coarsen(x, maxlen=200):
        cc = int(np.max(x.shape)/maxlen)
        return da.coarsen(np.mean, x, {0:cc, 1:cc}, trim_excess=True)
    
    Sc = da_coarsen(C.argmax(axis=0), 2000).compute()
    plt.imsave('script_out.jpg', label2rgb(Sc, bg_label=0))