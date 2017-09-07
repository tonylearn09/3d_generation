import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
import scipy.io as io
import numpy as np
from skimage import measure

try:
    from mpl_toolkits import mplot3d
    import trimesh
    from stl import mesh
except:
    print('Some function disabled')

def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

def plot(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f =  measure.marching_cubes_classic(voxels, level=threshold)
    #v, f =  measure.marching_cubes_classic(voxels)
    return v, f

