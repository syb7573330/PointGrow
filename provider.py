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
  """ Shuffle the order of point clouds and their labels
  """
  indices = range(len(points))
  shuffle(indices)
  return points[indices], labels[indices]

def shuffleConditionData(points, conditions, labels):
  """ Shuffle the order of point clouds, their conditions and labels
  """
  indices = range(len(points))
  shuffle(indices)
  return points[indices], conditions[indices], labels[indices]

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


##################################################
######### For farthest point sampling ############
##################################################
def calTriangleArea(p1, p2, p3):
  """   Calculate the area of the given triangle (p1, p2, p3)
  """
  p1p2 = np.array(p2) - np.array(p1)
  p1p3 = np.array(p3) - np.array(p1)
  return 0.5 * np.linalg.norm(np.cross(p1p2, p1p3))

def samplePoint(p1, p2, p3):
  """ p1, p2, p3: list of (3,) 
  """
  r1 = random()
  r2 = random()
  coef_p1 = 1.0 - math.sqrt(r1)
  coef_p2 = math.sqrt(r1)*(1.0-r2)
  coef_p3 = math.sqrt(r1)*r2
  x = p1[0] * coef_p1 + p2[0] * coef_p2 + p3[0] * coef_p3
  y = p1[1] * coef_p1 + p2[1] * coef_p2 + p3[1] * coef_p3
  z = p1[2] * coef_p1 + p2[2] * coef_p2 + p3[2] * coef_p3
  return x, y, z

def calc_distances(p0, points):
  return ((p0 - points)**2).sum(axis=1)

def farthestPointSampler(pts, K):
  """ pts: (num_pt, 3)
      K: an int to indicate the number of sampled points
  """
  # the distances between each sampled point to all the points
  sample_distances = []

  farthest_pts = np.zeros((K, 3))
  farthest_pts[0] = pts[0]
  distances = calc_distances(farthest_pts[0], pts)
  sample_distances.append(distances)

  for i in range(1, K):
    farthest_pts[i] = pts[np.argmax(distances)]
    distances_i = calc_distances(farthest_pts[i], pts)
    sample_distances.append(distances_i)
    distances = np.minimum(distances, distances_i)
  return farthest_pts, sample_distances
##################################################
##################################################


  