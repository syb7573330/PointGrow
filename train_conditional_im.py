import tensorflow as tf
import numpy as np
import argparse
import os
from random import shuffle
import glob
import time
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import conditional_model_im as MODEL
import provider
import im_util

parser = argparse.ArgumentParser()
parser.add_argument('--cat', default='02691156', help='The ShapeNet category')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--im_size', type=int, default=128, help='The image size [default: 128]')
parser.add_argument('--im_chan', type=int, default=4, help='The image channels [default: 4 w/ alpha]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 250]')
parser.add_argument('--num_output', type=int, default=200, help='The number of discrete coordinates per dimension [default: 200]')
parser.add_argument('--batch_size', type=int, default=15, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

CAT = FLAGS.cat
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = os.path.join(FLAGS.log_dir, "conditional_model_im", CAT)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % ("models/conditional_model_im.py", LOG_DIR)) # bkp of model def
os.system('cp train_conditional_im.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

NUM_OUTPUT = FLAGS.num_output
VOL_DIM = NUM_OUTPUT

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

IM_SIZE = FLAGS.im_size
IM_CHAN = FLAGS.im_chan

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)

def get_learning_rate(batch):
  learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            DECAY_STEP,          # Decay step.
            DECAY_RATE,          # Decay rate.
            staircase=True)
  learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
  return learning_rate

def get_bn_decay(batch):
  bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    batch*BATCH_SIZE,
                    BN_DECAY_DECAY_STEP,
                    BN_DECAY_DECAY_RATE,
                    staircase=True)
  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
  return bn_decay 

def train():
  with tf.Graph().as_default():
    with tf.device('/gpu:'+str(GPU_INDEX)):
      pointclouds_pl, imgs_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, IM_SIZE, IM_CHAN)
      is_train_pl = tf.placeholder(tf.bool)
      
      batch = tf.Variable(0, trainable=False)
      bn_decay = get_bn_decay(batch)
      learning_rate = get_learning_rate(batch)

      pred = MODEL.get_model(pointclouds_pl, imgs_pl, NUM_OUTPUT, is_train_pl, bn_decay=bn_decay)
      loss = MODEL.get_loss(pred, labels_pl)

      optimizer = tf.train.RMSPropOptimizer(learning_rate) # tf.train.RMSPropOptimizer(learning_rate)
      train_op = optimizer.minimize(loss, global_step=batch)

      saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      # Load training data into memeory
      # 1. Load point clouds
      points_train = provider.loadPC("./data/ShapeNet7/{}_train.npy".format(CAT), NUM_POINT)
      labels_train = provider.voxelizeData(points_train, VOL_DIM) # int, ranging from 0 to VOL_DIM-1
      points_train = labels_train / float(VOL_DIM) # fload, ranging from 0.0 to 1.0
      points_train = points_train.astype(np.float32)
      # 2. Load images
      im_ids_train = []
      with open("./data/ShapeNetRenderings/ids/{}_train_ids.txt".format(CAT), "r") as file_:
        im_ids_train = file_.readlines()
      im_ids_train = [line.rstrip() for line in im_ids_train]
      ims_train = []
      for id_ in im_ids_train:
        im_path_id = os.path.join("./data/ShapeNetRenderings/renderings", CAT, id_, 'rendering/00.png')
        ims_train.append(np.expand_dims(im_util.imread(im_path_id, [IM_SIZE, IM_SIZE]), axis=0))
      ims_train = np.concatenate(ims_train, axis=0)

      assert len(ims_train) == len(points_train)
      print ("All data loaded.")

      for epoch in range(MAX_EPOCH):
        # For training 
        log_string('----' + str(epoch) + '-----')
        current_data, current_conditions, current_label = provider.shuffleConditionData(points_train, ims_train, labels_train)
        print (current_data.shape, current_conditions.shape, current_label.shape)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        start_time = time.time()
        loss_sum = 0.0
        run_batches = 0
        for batch_idx in range(num_batches):
          start_idx = batch_idx * BATCH_SIZE
          end_idx = (batch_idx+1) * BATCH_SIZE

          feed_dict = {pointclouds_pl: current_data[start_idx:end_idx], 
                      imgs_pl: current_conditions[start_idx:end_idx],
                      labels_pl: current_label[start_idx:end_idx],
                      is_train_pl: True}
          step, _, loss_val = sess.run([batch, train_op, loss], feed_dict=feed_dict)

          loss_sum += loss_val
          run_batches += 1

        log_string('train mean loss: %f' % (loss_sum / float(run_batches)))
        print("train running time ", time.time() - start_time)

        if epoch % 5 == 0:
          save_path = saver.save(sess, os.path.join(LOG_DIR, 'model_' + str(epoch)+'.ckpt'))
          log_string("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  train()


