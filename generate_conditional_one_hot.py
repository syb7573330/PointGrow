""" This script generates point clouds for conditional PointGrow based on one-hot categorical vectors.
    We first generate 1050 points, and then use furthest point sampling methods 
    to sample 1024 from them.
  
  ShapeNet id       One-hot nonzero pos.
    '02691156'   ->        0
    '02958343'   ->        1
    '04379243'   ->        2
    '03001627'   ->        3 
    '02828884'   ->        4
    '02933112'   ->        5
    '03636649'   ->        6
"""
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
import conditional_model_one_hot as MODEL
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--cat_idx', type=int, default=0, help='The categorical index to conditionally generate point clouds [from 0 to 6].')
parser.add_argument('--cond_dim', type=int, default=7, help='The dimension of one-hot vector.')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_sampled_points', type=int, default=1050, help='The number of points to be sampled [default: 1050]')
parser.add_argument('--num_desired_points', type=int, default=1024, help='The desired number of points [default: 1024]')
parser.add_argument('--num_output', type=int, default=200, help='The number of discrete coordinates per dimension [default: 200]')
parser.add_argument('--tot_pc', type=int, default=20, help='The total number point clouds to generate.')
parser.add_argument('--batch_size', type=int, default=25, help='Batch Size during training [default: 32]')
FLAGS = parser.parse_args()

CAT_IDX = FLAGS.cat_idx
assert (CAT_IDX >= 0 and CAT_IDX <= 6)
TOTAL_PC = FLAGS.tot_pc
BATCH_SIZE = FLAGS.batch_size
NUM_SAMPLE_POINT = FLAGS.num_sampled_points
NUM_POINT = FLAGS.num_desired_points
GPU_INDEX = FLAGS.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
NUM_OUTPUT = FLAGS.num_output
VOL_DIM = NUM_OUTPUT 
COND_DIM = FLAGS.cond_dim

LOG_DIR = os.path.join(FLAGS.log_dir, "conditional_model_one_hot")

# Create a dictionary w/ keys being CAT_IDX and values being ShapeNet ids
cat_idx_dict = dict()
cat_idx_dict[0] = "02691156"
cat_idx_dict[1] = "02958343"
cat_idx_dict[2] = "04379243"
cat_idx_dict[3] = "03001627"
cat_idx_dict[4] = "02828884"
cat_idx_dict[5] = "02933112"
cat_idx_dict[6] = "03636649"

def generatePointClouds():
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, condition_pl, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_SAMPLE_POINT, COND_DIM)
      pred = MODEL.get_model(pointclouds_pl, condition_pl, NUM_OUTPUT)
      pred = tf.nn.softmax(pred)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    
    with tf.Session(config=config) as sess:
      model_path = LOG_DIR
      if tf.gfile.Exists(os.path.join(model_path, "checkpoint")):
        ckpt = tf.train.get_checkpoint_state(model_path)
        restorer = tf.train.Saver()
        restorer.restore(sess, ckpt.model_checkpoint_path)
        print ("Load parameters from checkpoint.")

      # Store final results
      results = []
      # Count valid generated shapes
      valid_count = 0
      # One-hot conditions
      conditions = np.zeros((BATCH_SIZE, COND_DIM))
      conditions[:, CAT_IDX] = 1
      while valid_count < TOTAL_PC:
        samples = np.zeros((BATCH_SIZE, NUM_SAMPLE_POINT, 3)).astype(np.float32)
        for pt_idx in range(NUM_SAMPLE_POINT):
          for coor_idx in range(3):
            feed_dict = {pointclouds_pl: samples, condition_pl: conditions}
            res = sess.run(pred, feed_dict=feed_dict) # (BATCH_SIZE, NUM_SAMPLE_POINT, 3, NUM_OUTPUT=200)
            to_be_sampled = res[:, pt_idx, coor_idx, :] # (BATCH_SIZE, NUM_OUTPUT=200)
            for batch_idx in range(BATCH_SIZE): # Sample for each point cloud within batch
              to_be_sampled_i = to_be_sampled[batch_idx] # (NUM_OUTPUT)
              to_be_sampled_i[to_be_sampled_i < (1.0/NUM_OUTPUT)] = 0 # Depress small probability
              to_be_sampled_i = to_be_sampled_i / np.sum(to_be_sampled_i) # Normalize the distribution
              res_i = np.random.choice(NUM_OUTPUT, 1, p=to_be_sampled_i)
              res_i = res_i[0] / float(NUM_OUTPUT)
              samples[batch_idx, pt_idx, coor_idx] = res_i
          print ("Category index: {}; generated point clouds: {}; generated points: {}".format(CAT_IDX, valid_count, pt_idx))
        
        # check valid point clouds from  'samples'
        valid_indices = []
        for index in range(BATCH_SIZE):
          sample_tmp = np.reshape(samples[index], [NUM_SAMPLE_POINT, 3]) # (z, y, x)
          vals, cnts = np.unique(sample_tmp, axis=0, return_counts=True)
          # Here we try two simple ways to filter point clouds with a considerable amount of overlapping.
          # Verification condition 1: More than 8 points share the same voxel.
          cond1 = np.max(cnts) >= 8
          # Verification condition 2: More than 3 voxels shared by more than 5 points.
          cond2 = np.sum(cnts > 5) >= 3
          if cond1 or cond2:
            print ("invlaid sample", index)
          else:
            valid_indices.append(index)

        for j in valid_indices:
          farthest_data, _ = provider.farthestPointSampler(samples[j], NUM_POINT)
          results.append(np.expand_dims(farthest_data, axis=0))
        valid_count += len(valid_indices)

      results = np.concatenate(results, axis=0)
      results = results[:TOTAL_PC]
      save_path = os.path.join("res", "conditional_model_one_hot")
      if not os.path.exists(save_path): os.makedirs(save_path)
      save_path = os.path.join(save_path, "res_{}.npy".format(cat_idx_dict[CAT_IDX]))
      np.save(save_path, results)
      print ("Results saved to {}.".format(save_path))

if __name__ == '__main__':
  generatePointClouds()






