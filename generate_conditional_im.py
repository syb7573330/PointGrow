""" This script generates point clouds for conditional PointGrow based on 2D image embeddings.
    We first generate 1050 points, and then use furthest point sampling methods to sample 1024 from them.
"""
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
import conditional_model_im as MODEL
import provider
import im_util

parser = argparse.ArgumentParser()
parser.add_argument('--cat', default='02691156', help='The ShapeNet category')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--im_size', type=int, default=128, help='The image size [default: 128]')
parser.add_argument('--im_chan', type=int, default=4, help='The image channels [default: 4 w/ alpha]')
parser.add_argument('--num_sampled_points', type=int, default=1050, help='The number of points to be sampled [default: 1050]')
parser.add_argument('--num_desired_points', type=int, default=1024, help='The desired number of points [default: 1024]')
parser.add_argument('--num_output', type=int, default=200, help='The number of discrete coordinates per dimension [default: 200]')
parser.add_argument('--tot_pc', type=int, default=20, help='The total number point clouds to generate.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 32]')
FLAGS = parser.parse_args()

CAT = FLAGS.cat
TOTAL_PC = FLAGS.tot_pc
BATCH_SIZE = FLAGS.batch_size
BATCH_SIZE = min(TOTAL_PC, BATCH_SIZE) # In case BATCH_SIZE is larger than TOTAL_PC.
TOTAL_PC = (TOTAL_PC // BATCH_SIZE) * BATCH_SIZE # Truncate TOTAL_PC

NUM_SAMPLE_POINT = FLAGS.num_sampled_points
NUM_POINT = FLAGS.num_desired_points
GPU_INDEX = FLAGS.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
NUM_OUTPUT = FLAGS.num_output
VOL_DIM = NUM_OUTPUT 
IM_SIZE = FLAGS.im_size
IM_CHAN = FLAGS.im_chan

LOG_DIR = os.path.join(FLAGS.log_dir, "conditional_model_im", CAT)

def generatePointClouds():
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, imgs_pl, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_SAMPLE_POINT, IM_SIZE, IM_CHAN)
      is_train_pl = tf.placeholder(tf.bool)
      pred = MODEL.get_model(pointclouds_pl, imgs_pl, NUM_OUTPUT, is_train_pl)
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

      # Load testing images
      im_ids_test = []
      with open("./data/ShapeNetRenderings/ids/{}_test_ids.txt".format(CAT), "r") as file_:
        im_ids_test = file_.readlines()
      im_ids_test = [line.rstrip() for line in im_ids_test]
      im_ids_test = im_ids_test[:TOTAL_PC]
      ims_test = []
      for id_ in im_ids_test:
        im_path_id = os.path.join("./data/ShapeNetRenderings/renderings", CAT, id_, 'rendering/00.png')
        ims_test.append(np.expand_dims(im_util.imread(im_path_id, [IM_SIZE, IM_SIZE]), axis=0))
      ims_test = np.concatenate(ims_test, axis=0)

      # Sampling point clouds
      results = []
      while len(results) < TOTAL_PC:
        samples = np.zeros((BATCH_SIZE, NUM_SAMPLE_POINT, 3)).astype(np.float32)
        for pt_idx in range(NUM_SAMPLE_POINT):
          for coor_idx in range(3):
            feed_dict = {pointclouds_pl: samples, 
                          imgs_pl: ims_test[len(results):len(results)+BATCH_SIZE], 
                          is_train_pl: False}
            res = sess.run(pred, feed_dict=feed_dict) # (BATCH_SIZE, NUM_SAMPLE_POINT, 3, NUM_OUTPUT=200)
            to_be_sampled = res[:, pt_idx, coor_idx, :] # (BATCH_SIZE, NUM_OUTPUT=200)
            for batch_idx in range(BATCH_SIZE): # Sample for each point cloud within batch
              to_be_sampled_i = to_be_sampled[batch_idx] # (NUM_OUTPUT)
              to_be_sampled_i[to_be_sampled_i < (1.0/NUM_OUTPUT)] = 0 # Depress small probability
              to_be_sampled_i = to_be_sampled_i / np.sum(to_be_sampled_i) # Normalize the distribution
              res_i = np.random.choice(NUM_OUTPUT, 1, p=to_be_sampled_i)
              res_i = res_i[0] / float(NUM_OUTPUT)
              samples[batch_idx, pt_idx, coor_idx] = res_i
          print ("Category: {}; generated point clouds: {}; generated points: {}".format(CAT, len(results), pt_idx))

        for j in range(BATCH_SIZE):
          farthest_data, _ = provider.farthestPointSampler(samples[j], NUM_POINT)
          results.append(np.expand_dims(farthest_data, axis=0))

      results = np.concatenate(results, axis=0)
      save_path = os.path.join("res", "conditional_model_im")
      if not os.path.exists(save_path): os.makedirs(save_path)
      save_path = os.path.join(save_path, "res_{}.npy".format(CAT))
      np.save(save_path, results)
      print ("Results saved to {}.".format(save_path))

if __name__ == '__main__':
  generatePointClouds()






