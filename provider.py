import os
import numpy as np
import sys
from random import shuffle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

## Download data if necessary
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'ShapeNet7')):
  www = 'https://www.dropbox.com/s/nlcswrxul1ymypw/ShapeNet7.zip'
  zipfile = os.path.basename(www)
  os.system('wget %s; unzip %s' % (www, zipfile))
  os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
  os.system('rm %s' % (zipfile))


def loadPC(file_path, num_pt):
  """ Load point cloud for a given file_path
  """
  data = np.load(file_path) # (n, num_point, 3)
  return data[:, :num_pt, :]

def shuffleData(points, labels):
  """ Shuffle the order of point clouds
  """
  indices = range(len(points))
  shuffle(indices)
  return points[indices], labels[indices]

def voxelizeData(points, vol_dim=200):
  """ voxelize point clouds
      Input: 
        points: (n, num_pt, 3) ranging from 0.0 to 1.0, ordered as (z, y, x)
        vol_dim: the number of bins to discretize point coordinates (or the dimension of a volume).
      Return:
        Voxelized point clouds, ranging from 0 to VOL_DIM-1
  """
  voxel = 1.0 / vol_dim
  points = points / voxel
  points[points>(vol_dim-1)] = vol_dim-1
  points[points<0] = 0
  points = points.astype(np.int32) # raning from [0, 199]
  return points




  