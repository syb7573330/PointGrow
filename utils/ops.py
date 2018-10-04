import numpy as np
import tensorflow as tf
from random import shuffle
  
WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()

def _variable_on_cpu(name, 
                     shape, 
                     initializer=WEIGHT_INITIALIZER,
                     trainable=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
  return var


def _variable_with_weight_decay(name, 
                                shape, 
                                wd=None,
                                initializer=WEIGHT_INITIALIZER,
                                trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape, initializer=initializer, trainable=trainable)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def get_weights(scope,
                weights_shape, 
                initializer=WEIGHT_INITIALIZER,
                weights_regularizer=None, 
                mask_type=None):
  """ Obtain weights for masked convolution operation.
  """
  kernel_h, kernel_w, channel, num_outputs = weights_shape
  weights = _variable_with_weight_decay(scope, weights_shape, wd=weights_regularizer, initializer=initializer)

  # Mask kernel
  if mask_type is not None:
      mask_type = mask_type.lower()

      # Mask 'a' for z
      mask = np.zeros(
        (kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)

      if mask_type == 'b': # Mask 'a' for y
        mask[:, :1, :, :] = 1.0
      elif mask_type == 'c': # Mask 'a' for x
        mask[:, :2, :, :] = 1.0

      weights *= tf.constant(mask, dtype=tf.float32)

  return weights

def conv2d(
    scope,
    inputs,
    num_outputs,
    kernel_shape,
    mask_type=None, 
    strides=[1, 1],
    padding="SAME",
    activation_fn=None,
    weights_initializer=WEIGHT_INITIALIZER,
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=False,
    bn=False,
    bn_decay=None,
    is_training=None):

  """ 2D convolution with potential masks.
    Args:
    inputs: 4-D tensor variable BxHxWxC
    num_outputs: Int
    kernel_shape: A list of 2 ints, [kernel_height, kernel_width]
    mask_type: None, "A", "B" or "C".
       Assuming all points are arranges as (z, y, x)
       'A', [o, o, o] for z coordinate
       'B', [x, o, o] for y coordinate
       'C', [x, x, o] for x coordinate
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  """

  with tf.variable_scope(scope, reuse=reuse):
    batch_size, height, width, channel = inputs.get_shape().as_list()
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides

    weights_shape = [kernel_h, kernel_w, channel, num_outputs]
    weights = get_weights("weights", weights_shape, 
                initializer=weights_initializer, 
                weights_regularizer=weights_regularizer, 
                mask_type=mask_type)
      
    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    
    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if bn:
      outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    return outputs

def conv2d_cond(
    scope,
    inputs,
    conditions,
    num_outputs,
    kernel_shape,
    mask_type=None,
    strides=[1, 1],
    padding="SAME",
    activation_fn=None,
    weights_initializer=WEIGHT_INITIALIZER,
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=False,
    bn=False,
    bn_decay=None,
    is_training=None):
  """ 2D conditional convolution with potential masks.
    Args:
    inputs: 4-D tensor variable BxHxWxC
    conditions: condition tensor of shape (batch, 1, 1, cond_dim)
    num_outputs: Int
    kernel_shape: A list of 2 ints, [kernel_height, kernel_width]
    mask_type: None, "A", "B" or "C".
       Assuming all points are arranges as (z, y, x)
       'A', [o, o, o] for z coordinate
       'B', [x, o, o] for y coordinate
       'C', [x, x, o] for x coordinate
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  """

  with tf.variable_scope(scope, reuse=reuse):
    batch_size, height, width, channel = inputs.get_shape().as_list()
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    weights_shape = [kernel_h, kernel_w, channel, num_outputs]
    weights = get_weights("weights", weights_shape, 
                initializer=weights_initializer, 
                weights_regularizer=weights_regularizer, 
                mask_type=mask_type)
    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    
    # condition
    cond_dim = conditions.get_shape()[-1].value
    weights_shape_cond = [1, 1, cond_dim, num_outputs]
    weights_cond = get_weights("condition_weight", weights_shape_cond, 
                initializer=weights_initializer)
    outputs_cond = tf.nn.conv2d(conditions,
        weights_cond, [1, 1, 1, 1], padding='VALID', name='outputs_cond')

    # total ouputs
    outputs = outputs + outputs_cond

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if bn:
      outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    return outputs

def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
  return outputs

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed

def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)


