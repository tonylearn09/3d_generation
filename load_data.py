import os
import sys

import scipy
import scipy.io as io
import numpy as np
import skimage.measure as sk

data_dir = '3DShapeNets/volumetric_data/'

def load_data(obj_class='chair', is_train=True):
    obj_path = os.path.join(data_dir, obj_class, '30')
    #obj_path = os.path.join(obj_class, '30')
    #obj_path = data_dir + obj_class + '/30/'
    if is_train:
        obj_path = os.path.join(obj_path, 'train')
        #obj_path += 'train/'
    else:
        obj_path = os.path.join(obj_path, 'test')
        #obj_path += 'test/'

    files = [f for f in os.listdir(obj_path) if f.endswith('.mat')]
    if is_train:
        #files = files[0: int(0.8*len(files))]
        volume_batch = np.asarray([get_voxel_from_mat(os.path.join(obj_path , f)) for f in files], dtype=np.bool)
    return volume_batch

def get_voxel_from_mat(path):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    voxels = scipy.ndimage.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels




