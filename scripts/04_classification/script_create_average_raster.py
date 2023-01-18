import src.utils as ut
import numpy as np
import matplotlib.pyplot as plt
import os

folder1 = 'v3_paalaki_RF_invF'
folder2 = 'v3_paalaki_resnet_invF'

prefix1 = 'paalaki_RF_invF'
prefix2 = 'v3_paalaki'

outpath = '../tulkinnat/luokittelut/v3_avg_invF/'

for n in range(26):
    R1, src1 = ut.read_raster_for_classification(f'../tulkinnat/luokittelut/{folder1}/scores/{prefix1}_{n}scores.tif')
    R2, src2 = ut.read_raster_for_classification(f'../tulkinnat/luokittelut/{folder2}/scores/{prefix2}_{n}_segscores.tif')
    d1 = R1.shape[0]-R2.shape[0]
    d2 = R1.shape[1]-R2.shape[1]
    
    R1 = R1[d1//2:-d1//2,d2//2:-d2//2,:]

    N_classes = R1.shape[2]

    M = (R1+R2)/2
    Mclass = np.argmax(M, axis=2)
    Mout = np.uint8(np.round(M))
    Mout = np.moveaxis(Mout, 2,0)

    ut.write_classified_raster(os.path.join(outpath, f'{prefix2}_{n}'), src2, Mclass)
    ut.write_float_raster(os.path.join(outpath, f'{prefix2}_{n}scores'), src2, Mout, N_classes)
    print(n)

plt.figure(figsize=(10,10))
plt.imshow(M)