# -*- coding:utf-8 -*-

from __future__ import print_function
import os
import time
from glob import glob
import sys

from ops import *

from utils import *


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, batch_size=64, sample_num=64, z_dim=100, gf_dim=64, df_dim=64, c_dim=2,
               dataset_name='default', checkpoint_dir=None, sample_dir=None):
    """
    :param sess: TensorFlow session
    :param batch_size: The size of batch for training.
    :param y_dim: Dimention of dim for y. [None]
    :param z_dim: Dimention of dim for z. [100]
    :param gf_dim: Dimention of generator filter in first conv layer. [64]
    :param df_dim: Dimention of discrimnator filter in first conv layer. [64]  
    """
    self.sess = sess

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.filenames = [os.path.join('../data/tfrecords/', self.dataset_name + '.tfrecords')]
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir

    self.c_dim = c_dim
    self.build_model()

  def build_model(self):
    self.real_inputs = get_inputs(input_height=self.input_height, input_width=self.input_width, c_dim=self.c_dim, batch_size=self.batch_size, filenames=self.filenames)
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary('z', self.z)

    self.G = self.generator(self.z)
    self.D, self.D_logits = self.discriminator(self.real_inputs)

    self.sample, self.sample_pool = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_sum = histogram_summary('d', self.D)
    self.d_sum_ = histogram_summary('d', self.D_)
    self.g_sum = histogram_summary('g', self.G)

    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary('d_loss_real', self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary('d_loss_fake', self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary('g_loss', self.g_loss)
    self.d_loss_sum = scalar_summary('d_loss', self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
    tf.global_variables_initializer().run()

    self.g_sum = merge_summary([self.z_sum, self.d_sum_, self.g_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter('./logs', self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)

    if could_load:
      counter = checkpoint_counter
      print('Load Success')
    else:
      print('Load Failed')

    for epoch in xrange(config.epoch):
      batch_num = config.train_size // config.batch_size

      for index in xrange(0, batch_num):
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        # 更新判别器
        _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.z: batch_z})
        self.writer.add_summary(summary_str, counter)

        #_, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={self.z: batch_z})
        #self.writer.add_summary(summary_str, counter)

        # 更新生成器
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z})
        self.writer.add_summary(summary_str, counter)

        # 再更新一次生成器，使得判别器的损失值不至于变为0
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z})
        self.writer.add_summary(summary_str, counter)

        # errD_fake = self.d_loss_fake.eval({self.z: batch_z})
        # errD_real = self.d_loss_real.eval()
        # errG = self.g_loss.eval({self.z: batch_z})
        errD_fake, errD_real, errG = self.sess.run([self.d_loss_fake, self.d_loss_real, self.g_loss], feed_dict={self.z: batch_z})
        counter += 1
        print('Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f' % (epoch, index, batch_num, time.time() - start_time, errD_fake + errD_real, errG))

        if np.mod(counter, 100) == 1:
          samples, d_loss, g_loss = self.sess.run([self.sample, self.d_loss, self.g_loss], feed_dict={self.z: sample_z})
          save_csv(samples, './{}/train_{:02d}_{:03d}.csv'.format(config.sample_dir, epoch, index), input_height=self.input_height, input_width=self.input_width, sample_num=self.sample_num)
          print('csv file has been saved')

        if np.mod(counter, 50) == 2:
          self.save(config.checkpoint_dir, counter)
    coord.request_stop()
    coord.join(threads)


  def generator(self, z):
    with tf.variable_scope('generator'):
      s_h, s_w = self.input_height, self.input_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_linear', with_w=True)
      self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      self.h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(self.h2))

      self.h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(self.h3))

      self.h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

      return tf.nn.tanh(self.h4)

  def discriminator(self, image, reuse=False):
    with tf.variable_scope('discriminator') as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 4, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_linear')
      return tf.nn.sigmoid(h4), h4

  def sampler(self, z):
    with tf.variable_scope('generator') as scope:
      scope.reuse_variables()

      s_h, s_w = self.input_height, self.input_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      h0 = tf.reshape(linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_linear'), [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, is_training=False))

      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, is_training=False))

      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, is_training=False))

      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, is_training=False))

      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
      h4 = tf.nn.tanh(h4)
      h4_pool = tf.layers.average_pooling2d(h4, pool_size=3, strides=1, padding='SAME')
      return h4, h4_pool

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(self.dataset_name, self.batch_size, self.input_height, self.input_width)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
