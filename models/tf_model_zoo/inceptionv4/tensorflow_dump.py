# python3

# TensorBoard
# python3 ~/.local/lib/python3.5/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=logs --port=6007

import os
import sys
import h5py
import math
import urllib.request
import numpy as np
import tensorflow as tf

sys.path.append('models/slim')
from datasets import dataset_utils
from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

image_size = inception.inception_v3.default_image_size

url = 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'
checkpoints_dir = '/tmp/checkpoints/'

def make_padding(padding_name, conv_shape):
  padding_name = padding_name.decode("utf-8")
  if padding_name == "VALID":
    return [0, 0]
  elif padding_name == "SAME":
    #return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
    return [math.floor(int(conv_shape[0])/2), math.floor(int(conv_shape[1])/2)]
  else:
    sys.exit('Invalid padding name '+padding_name)

def dump_conv2d(name='Conv2d_1a_3x3'):
  
  conv_operation = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/'+name+'/Conv2D') # remplacer convolution par Conv2D si erreur

  weights_tensor = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/weights:0')
  weights = weights_tensor.eval()

  padding = make_padding(conv_operation.get_attr('padding'), weights_tensor.get_shape())
  strides = conv_operation.get_attr('strides')

  conv_out = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/'+name+'/Conv2D').outputs[0].eval() # remplacer convolution par Conv2D si erreur
  
  beta = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/beta:0').eval()
  #gamma = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/gamma:0').eval()
  mean = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/moving_mean:0').eval()
  var = sess.graph.get_tensor_by_name('InceptionV4/'+name+'/BatchNorm/moving_variance:0').eval()
  
  relu_out = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/'+name+'/Relu').outputs[0].eval()

  os.system('mkdir -p dump/InceptionV4/'+name)
  h5f = h5py.File('dump/InceptionV4/'+name+'.h5', 'w')
  # conv
  h5f.create_dataset("weights", data=weights)
  h5f.create_dataset("strides", data=strides)
  h5f.create_dataset("padding", data=padding)
  h5f.create_dataset("conv_out", data=conv_out)
  # batch norm
  h5f.create_dataset("beta", data=beta)
  #h5f.create_dataset("gamma", data=gamma)
  h5f.create_dataset("mean", data=mean)
  h5f.create_dataset("var", data=var)
  h5f.create_dataset("relu_out", data=relu_out)
  h5f.close()

def dump_logits():
  operation = sess.graph.get_operation_by_name('InceptionV4/Logits/Predictions')

  weights_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Logits/weights:0')
  weights = weights_tensor.eval()

  biases_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Logits/biases:0')
  biases = biases_tensor.eval()

  out = operation.outputs[0].eval()

  h5f = h5py.File('dump/InceptionV4/Logits.h5', 'w')
  # conv
  h5f.create_dataset("weights", data=weights)
  h5f.create_dataset("biases", data=biases)
  h5f.create_dataset("out", data=out)
  h5f.close()


# def dump_avgpool(name='Mixed_5b/Branch_3/AvgPool_0a_3x3'):
#   operation = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/'+name+'/AvgPool')
#   out = operation.outputs[0].eval()
#   os.system('mkdir -p dump/InceptionV4/'+name)
#   h5f = h5py.File('dump/InceptionV4/'+name+'.h5', 'w')
#   h5f.create_dataset("out", data=out)
#   h5f.close()

# def dump_concats():
#   operation1 = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/Mixed_7b/Branch_1/concat')
#   operation2 = sess.graph.get_operation_by_name('InceptionV4/InceptionV4/Mixed_7b/Branch_2/concat')
#   out1 = operation1.outputs[0].eval()
#   out2 = operation2.outputs[0].eval()
#   os.system('mkdir -p dump/InceptionV4/Mixed_7b/Branch_1/concat')
#   h5f = h5py.File('dump/InceptionV4/Mixed_7b/Branch_1/concat.h5', 'w')
#   h5f.create_dataset("out", data=out1)
#   h5f.close()
#   os.system('mkdir -p dump/InceptionV4/Mixed_7b/Branch_2/concat')
#   h5f = h5py.File('dump/InceptionV4/Mixed_7b/Branch_2/concat.h5', 'w')
#   h5f.create_dataset("out", data=out2)
#   h5f.close()

def dump_mixed_4a_7a(name='Mixed_4a'):
  dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_0/Conv2d_1a_3x3')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0b_1x7')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0c_7x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_1a_3x3')

def dump_mixed_5(name='Mixed_5b'):
  dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0b_3x3')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0b_3x3')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0c_3x3')
  dump_conv2d(name=name+'/Branch_3/Conv2d_0b_1x1')

def dump_mixed_6(name='Mixed_6b'):
  dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0b_1x7')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0c_7x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0b_7x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0c_1x7')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0d_7x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0e_1x7')
  dump_conv2d(name=name+'/Branch_3/Conv2d_0b_1x1')

def dump_mixed_7(name='Mixed_7b'):
  dump_conv2d(name=name+'/Branch_0/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0b_1x3')
  dump_conv2d(name=name+'/Branch_1/Conv2d_0c_3x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0a_1x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0b_3x1')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0c_1x3')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0d_1x3')
  dump_conv2d(name=name+'/Branch_2/Conv2d_0e_3x1')
  dump_conv2d(name=name+'/Branch_3/Conv2d_0b_1x1')


if not tf.gfile.Exists(checkpoints_dir+'inception_v4.ckpt'):
  tf.gfile.MakeDirs(checkpoints_dir)
  dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

with tf.Graph().as_default():

  # Create model architecture

  from scipy import misc
  img = misc.imread('lena_299.png')
  print(img.shape)

  inputs = np.zeros((1,299,299,3), dtype=np.float32)

  inputs[0] = img
  inputs = tf.pack(inputs)

  with slim.arg_scope(inception.inception_v4_arg_scope()):
    logits, _ = inception.inception_v4(inputs, num_classes=1001, is_training=False)

  with tf.Session() as sess:

    # Initialize model
    init_fn = slim.assign_from_checkpoint_fn(
    os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
    slim.get_model_variables('InceptionV4'))  

    init_fn(sess)

    # Display model variables
    for v in slim.get_model_variables():
      print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Create graph
    os.system("rm -rf logs")
    os.system("mkdir -p logs")

    tf.scalar_summary('logs', logits[0][0])
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter("logs", sess.graph)

    out = sess.run(summary_op)
    summary_writer.add_summary(out, 0)

    # Stem
    dump_conv2d(name='Conv2d_1a_3x3')
    dump_conv2d(name='Conv2d_2a_3x3')
    dump_conv2d(name='Conv2d_2b_3x3')

    dump_conv2d(name='Mixed_3a/Branch_1/Conv2d_0a_3x3')
    dump_mixed_4a_7a(name='Mixed_4a')
    dump_conv2d(name='Mixed_5a/Branch_0/Conv2d_1a_3x3')

    # Inception A
    dump_mixed_5(name='Mixed_5b')
    dump_mixed_5(name='Mixed_5c')
    dump_mixed_5(name='Mixed_5d')
    dump_mixed_5(name='Mixed_5e')

    # Reduction A
    dump_conv2d(name='Mixed_6a/Branch_0/Conv2d_1a_3x3')
    dump_conv2d(name='Mixed_6a/Branch_1/Conv2d_0a_1x1')
    dump_conv2d(name='Mixed_6a/Branch_1/Conv2d_0b_3x3')
    dump_conv2d(name='Mixed_6a/Branch_1/Conv2d_1a_3x3')

    # Inception B
    dump_mixed_6(name='Mixed_6b')
    dump_mixed_6(name='Mixed_6c')
    dump_mixed_6(name='Mixed_6d')
    dump_mixed_6(name='Mixed_6e')
    dump_mixed_6(name='Mixed_6f')
    dump_mixed_6(name='Mixed_6g')
    dump_mixed_6(name='Mixed_6h')

    # Reduction B
    dump_mixed_4a_7a(name='Mixed_7a')

    # Inception C
    dump_mixed_7(name='Mixed_7b')
    dump_mixed_7(name='Mixed_7c')
    dump_mixed_7(name='Mixed_7d')

    dump_logits()


    # #dump_concats()
    # #dump_avgpool(name='Mixed_5b/Branch_3/AvgPool_0a_3x3')