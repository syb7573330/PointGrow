import numpy as np
from random import *
import math
import sys
sys.path.append("..")
import provider

def sample_pc_from_obj(path, n_samples=5000, furthest_samples=1024):
  """ Sample a point cloud from the path to an .obj mesh file.
      Inputs:
        path: The path to an .obj file.
        n_samples: The total number of points to sample from the mesh.
        furthest_samples: The final number of points to keep representing the shapes using farthest point sampling.
      Return:
        A numpy array of size (furthest_samples, 3) representing the sampled point cloud, ordered as (x, y, z).
  """
  # Store vertices and faces
  vs = []
  fs = []
  lines = []
  # read lines
  with open(path, 'r') as file_:
    lines = file_.readlines()
  # read vertices and faces
  for line in lines:
    if len(line) > 3 and line.startswith('v '):
      line_ = line.rstrip().split()
      vs.append([float(line_[1]), float(line_[2]), float(line_[3])])
    if len(line) > 3 and line.startswith('f '):
      line_ = line.rstrip().split()
      v1 = int(line_[1].split('/')[0]) - 1
      v2 = int(line_[2].split('/')[0]) - 1
      v3 = int(line_[3].split('/')[0]) - 1
      fs.append([v1, v2, v3])

  # accumulate face areas
  accumulatedAreas = [0.0] * len(fs)
  for i in range(len(fs)):
    p1 = vs[fs[i][0]]
    p2 = vs[fs[i][1]]
    p3 = vs[fs[i][2]]
    area = provider.calTriangleArea(p1, p2, p3)
    if i == 0:
      accumulatedAreas[i] = area
    else:
      accumulatedAreas[i] = accumulatedAreas[i-1] + area

  # sample points
  sampled_pts = np.zeros((n_samples, 3))
  for i in range(n_samples):
    # generate a number within accumulatedAreas[-1]
    randomNumber = random() * accumulatedAreas[-1]
    index = np.searchsorted(accumulatedAreas, randomNumber)
    p1 = vs[fs[index][0]]
    p2 = vs[fs[index][1]]
    p3 = vs[fs[index][2]]
    x, y, z = provider.samplePoint(p1, p2, p3)
    sampled_pts[i, 0] = x
    sampled_pts[i, 1] = y
    sampled_pts[i, 2] = z

  farthest_pts, _ = farthestPointSampler(sampled_pts, furthest_samples)
  return farthest_pts
