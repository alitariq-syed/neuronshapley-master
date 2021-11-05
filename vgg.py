# coding=utf-8
# coding=utf-8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import os
import socket

from tf_slim import layers

from tf_slim.layers import layers as layers_lib
from tf_slim.layers import regularizers
from tf_slim.layers import utils
from tf_slim.ops.arg_scope import arg_scope

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
# pylint:enable=g-direct-tensorflow-import


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      activation_fn=nn_ops.relu,
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      biases_initializer=init_ops.zeros_initializer()):
    with arg_scope([layers.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 1, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 1, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 2, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 2, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 2, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with variable_scope.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
        outputs_collections=end_points_collection):
      net = layers_lib.repeat(
          inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
      net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
      net = layers_lib.repeat(net, 4, layers.conv2d, 256, [3, 3], scope='conv3')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
      net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv4')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
      net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv5')
      net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
      net = layers_lib.dropout(
          net, dropout_keep_prob, is_training=is_training, scope='dropout7')
      net = layers.conv2d(
          net,
          num_classes, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points


vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19


class vgg16(object):
    
    def __init__(self, sess, saver, input_tensor, labels_tensor, end_points, labels_path):
        
        ENDS_DIC = {'Predictions':'prediction',
                   'Logits':'logits'}
        self.labels = tf.gfile.Open(labels_path).read().splitlines()
        self.y_input = labels_tensor
        self.bottlenecks_tensors={}

        self.ends = end_points
        #self.starts = start_points
        self.ends.update({'input': input_tensor})
        for key in ENDS_DIC:
            self.ends[ENDS_DIC[key]] = end_points[key]
        self.logits = self.ends['logits']
        self.prelogits = self.ends['prelogits']
        self.input = self.ends['input']
        self.probs = end_points['Predictions']
        self.sess = sess
        self.saver = saver
        self.model_name = 'vgg16'
        self.sample_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_input, len(self.labels)),
            logits=self.ends['logits'])
        self.loss = tf.reduce_mean(self.sample_loss)
        self.sample_accuracy = tf.cast(
            tf.math.equal(tf.math.argmax(self.ends['logits'], -1), 
                          self.y_input), 
            tf.float32
        )
        self.accuracy = tf.reduce_mean(self.sample_accuracy)
        self.image_shape = (299, 299)        
        
#    def _make_gradient_tensors(self):
#        """Makes gradient tensors for all bottleneck tensors.
#        """
#        self.bottlenecks_gradients = {}
#        for bn in self.bottlenecks_tensors:
#            self.bottlenecks_gradients[bn] = tf.gradients(
#                self.loss, self.bottlenecks_tensors[bn])[0]

#    def get_gradient(self, acts, y, bottleneck_name):
        
#        return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
#                self.bottlenecks_tensors[bottleneck_name]: acts,
#                self.y_input: y
#        })
    
    def get_predictions(self, imgs):
        
        return  self.sess.run(self.ends['prediction'], {self.ends['input']: imgs})

    def run_imgs(self, imgs, bottleneck_name):
        
        return self.sess.run(self.bottlenecks_tensors[bottleneck_name],
                                                 {self.ends['input']: imgs})
    
    def reshape_activations(self, layer_acts):
        
        return np.asarray(layer_acts).squeeze()

    def id_to_label(self, idx):
        return self.labels[idx]
    
    def label_to_id(self, label):
        return self.labels.index(label)
    
    def restore(self, checkpoint):
        self.saver.restore(self.sess, checkpoint)


def vgg_instance(input_tensor=None, checkpoint=None, labels_path=None, sess=None):
    
    if sess is None:
        config = tf.ConfigProto()
        if 'arthur' in socket.gethostname():
            #config.gpu_options.per_process_gpu_memory_fraction = 0.45
            config.gpu_options.allow_growth=True
        else:
            config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
    arg_scope = vgg_arg_scope()
    if input_tensor is None:
        input_tensor = tf.placeholder(tf.float32, (None, 299, 299, 3), name='Input')
    labels_tensor = tf.placeholder(tf.int64, (None), name='Label')
    with slim.arg_scope(arg_scope):
        logits, end_points = vgg_16((input_tensor - 0.5) * 2, 1001, is_training=False)
        model_variables = tf.global_variables(scope='vgg16')
        saver = tf.train.Saver(var_list=model_variables)
        if labels_path is None:
            labels_path = os.path.join('imagenet_labels.txt')
        model = vgg16(sess, saver, input_tensor, labels_tensor, end_points, 
                                           labels_path)
        if checkpoint is None:
            model.sess.run(tf.initializers.variables(model_variables))
        else:
            model.restore(checkpoint)
    return model
