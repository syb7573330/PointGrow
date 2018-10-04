""" The conditional PointGrow with one-hot condition
"""
import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import ops

def placeholder_inputs(batch_size, num_point, cond_dim):
  """ Return point clouds placeholder, one-hot condition placeholder and their discrete coordinates
  """
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  condition_pl = tf.placeholder(tf.float32, shape=(batch_size, cond_dim))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point, 3)) 
  return pointclouds_pl, condition_pl, labels_pl

def contextAwarenessOp(name, ftr_ten):
  """ inputs: 
        ftr_ten: feature tensor of size (batch_size, num_point, 1, ftr_dim)
      return: (batch_size, num_point, 1, ftr_dim)
  """
  num_point = ftr_ten.get_shape()[1].value
  with tf.variable_scope(name):
    net_context = []
    for i in range(num_point):
      contxt_tmp = tf.reduce_mean(ftr_ten[:, :i+1, :, :], axis=1, keepdims=True)
      net_context.append(contxt_tmp)
    net_context = tf.concat(net_context, axis=1)
  return net_context

def get_model(inputs, conditions, output_dim, wd=None):
  """ inputs: (batch_size, num_point, 3)
      conditions: One-hot conditions (batch_size, cond_dim)
      returns: (batch_size, num_point, 3, output_dim)
  """
  batch_size = inputs.get_shape()[0].value
  num_point = inputs.get_shape()[1].value
  cond_dim = conditions.get_shape()[1].value

  inputs = tf.expand_dims(inputs, -1) # (batch_size, num_point, 3, 1)
  conditions = tf.reshape(conditions, (batch_size, 1, 1, cond_dim))

  #####################################
  net = ops.conv2d_cond("conv1", inputs, conditions, 64,
    kernel_shape=[1, 3], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)

  net = ops.conv2d_cond("conv2", net, conditions, 64,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)

  net = ops.conv2d_cond("conv3", net, conditions, 128,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_context = contextAwarenessOp("context3", net)
  
  net = tf.concat([net, net_context], axis=-1) # (batch, num_pt, 1, ftr_dim3)

  embed_ftr_dim = 128
  net = ops.conv2d_cond("conv4", net, conditions, embed_ftr_dim,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net = ops.conv2d_cond("conv5", net, conditions, embed_ftr_dim,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  
  ########## Self-Attention Context Awareness operation ##########
  net_context = contextAwarenessOp("context_saca", net)
  confidence_input = tf.concat([net, net_context], axis=-1)

  confidence = ops.conv2d_cond('conf1', confidence_input, conditions, embed_ftr_dim,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  confidence = ops.conv2d_cond('conf2', confidence, conditions, embed_ftr_dim,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  confidence = ops.conv2d_cond('conf3', confidence, conditions, embed_ftr_dim,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)

  net_conf = tf.multiply(net, confidence) # [batch_size, num_point, 1, embed_ftr_dim]

  initial_context = tf.zeros([batch_size, 1, 1, embed_ftr_dim])
  net_context = [initial_context]
  for i in range(num_point-1):
    net_context.append(net_context[-1] + net_conf[:, i:i+1, :, :])
  net_context = tf.concat(net_context, axis=1) # [batch_size, num_point, 1, 128]
  ###############################################################

  # For first dimension (z)
  net_pc_1 = ops.conv2d_cond("pc_dim_1_0", net_context, conditions, 128,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_1 = ops.conv2d_cond("pc_dim_1_1", net_pc_1, conditions, 128,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_1 = ops.conv2d_cond("pc_dim_1_2", net_pc_1, conditions, output_dim,
    kernel_shape=[1, 1], strides=[1, 1], padding="VALID")

  # For second dimension (y)
  net_pc_2 = ops.conv2d_cond("pc_coord_2_0", inputs, conditions, 32,
    kernel_shape=[1, 3], mask_type='B', activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_2 = ops.conv2d_cond("pc_coord_2_1", net_pc_2, conditions, 32,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_2 = tf.concat([net_context, net_pc_2], axis=-1)
  net_pc_2 = ops.conv2d_cond("pc_dim_2_0", net_pc_2, conditions, 128,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_2 = ops.conv2d_cond("pc_dim_2_1", net_pc_2, conditions, 128,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_2 = ops.conv2d_cond("pc_dim_2_2", net_pc_2, conditions, output_dim,
    kernel_shape=[1, 1], strides=[1, 1], padding="VALID")

  # For third dimension (x)
  net_pc_3 = ops.conv2d_cond("pc_coord_3_0", inputs, conditions, 32,
    kernel_shape=[1, 3], mask_type='C', activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_3 = ops.conv2d_cond("pc_coord_3_1", net_pc_3, conditions, 32,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_3 = tf.concat([net_context, net_pc_3], axis=-1)
  net_pc_3 = ops.conv2d_cond("pc_dim_3_0", net_pc_3, conditions, 128,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_3 = ops.conv2d_cond("pc_dim_3_1", net_pc_3, conditions, 128,
    kernel_shape=[1, 1], activation_fn=tf.nn.elu,
    strides=[1, 1], padding="VALID", weights_regularizer=wd)
  net_pc_3 = ops.conv2d_cond("pc_dim_3_2", net_pc_3, conditions, output_dim,
    kernel_shape=[1, 1], strides=[1, 1], padding="VALID")

  return tf.concat([net_pc_1, net_pc_2, net_pc_3], axis=2)

def get_loss(logits, targets):
  """ logits: (batch, num_pt, 3, ouput_dim)
      targets: (batch, num_pt, 3)
  """
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=targets))
  tf.add_to_collection('losses', loss)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


if __name__ == '__main__':
  batch_size = 10 
  num_point = 124
  output_dim = 20
  cond_dim = 7


  pointclouds_pl, condition_pl, labels_pl = placeholder_inputs(batch_size, num_point, cond_dim)
  logits = get_model(pointclouds_pl, condition_pl, output_dim, wd=None)
  loss = get_loss(logits, labels_pl)