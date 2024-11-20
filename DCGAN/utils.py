# -*- coding:utf-8 -*-
from __future__ import print_function
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim
import moviepy.editor as mpy
import csv
import os

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w * k_h * x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def inverse_transform(images):
  return (images+1.)/2.

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width, resize_height, resize_width, crop)

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def imread(path, grayscale=False):
  if (grayscale):
    return scipy.misc.imread(path, flatten=True).astype(np.float32)
  else:
    return scipy.misc.imread(path).astype(np.float32)

# def get_tensor(csv_path, input_height, input_width):
#   input_tensor = np.zeros((input_height, input_width, 2)).astype(np.float32)
#   with open(csv_path, 'rb') as f:
#     reader = csv.reader(f)
#     for line in reader:
#       if input_tensor[int(line[0])][int(line[1])][0] == 0.:
#         input_tensor[int(line[0])][int(line[1])][0] = float(line[2])
#       else:
#         if input_tensor[int(line[0])][int(line[1])][0] < float(line[2]):
#           input_tensor[int(line[0])][int(line[1])][1] = float(line[2])
#         else:
#           input_tensor[int(line[0])][int(line[1])][1] = input_tensor[int(line[0])][int(line[1])][0]
#           input_tensor[int(line[0])][int(line[1])][0] = float(line[2])
#   return input_tensor / 127.5 - 1

def save_csv(tensor, csv_path, input_height, input_width, sample_num):
  tensor_new = (tensor + 1.) * 127.5 
  with open(csv_path, 'wb') as f:
    writer = csv.writer(f)
    i = random.randint(0, sample_num-1)
    for x in range(input_height):
      for y in range(input_width):
        if (tensor_new[i][x][y][0] > 0.0) and (tensor_new[i][x][y][1] > 0.0):
          writer.writerow([float(x), float(y), tensor_new[i][x][y][0]])
          writer.writerow([float(x), float(y), tensor_new[i][x][y][1]])
        else:
          continue

def read_and_decode(filename_queue, input_height, input_width, c_dim):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features={'img_raw': tf.FixedLenFeature([], tf.string)})
  tensor = tf.decode_raw(features['img_raw'], tf.int32)
  tensor = tf.reshape(tensor, [input_height, input_width, c_dim])
  tensor = tf.cast(tensor, tf.float32)
  tensor = tensor / 127.5 - 1.
  return tensor

def get_inputs(input_height, input_width, c_dim, batch_size, filenames):
  filename_queue = tf.train.string_input_producer(filenames)
  tensor = read_and_decode(filename_queue, input_height, input_width, c_dim)
  input_batch = tf.train.shuffle_batch([tensor], batch_size=batch_size, num_threads=4, capacity=20000+3*batch_size, min_after_dequeue=20000)
  return input_batch
