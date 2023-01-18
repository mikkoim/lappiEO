import src.utils as ut
import numpy as np
import matplotlib.pyplot as plt

folder = 'v3_saariselka_RF_invF'
fileprefix = 'v3_saariselka_RF_invF'
for n in range(19):
    R, src = ut.read_raster_for_classification(f'../tulkinnat/luokittelut/{folder}/scores/{fileprefix}_{n}scores.tif')

    M = np.max(R,axis=2)
    ut.write_classified_raster(f'../tulkinnat/luokittelut/{folder}/confidence/{fileprefix}_{n}_conf', src, M)
    print(n)

plt.figure(figsize=(10,10))
plt.imshow(M)