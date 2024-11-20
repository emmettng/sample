# -*- coding:utf-8 -*-
import os
from model import DCGAN
from utils import pp

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 96, "The number of input images [96]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("c_dim", 2, "The channel of the input picture. If grayscale the value is set to 1. [2]")
flags.DEFINE_string("dataset", "train75", "The filename of tfrecords [train75]")
# flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "logs", "Directory name to save the checkpoints [logs]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
  # 打印参数
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.mkdir(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.mkdir(FLAGS.sample_dir)

  # 分配 GPU 资源
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth = True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(sess=sess,
                  input_width=FLAGS.input_width,
                  input_height=FLAGS.input_height,
                  batch_size=FLAGS.batch_size,
                  sample_num=FLAGS.batch_size,
                  dataset_name=FLAGS.dataset,
                  c_dim=FLAGS.c_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception('Train a model first, then run the test mode')

if __name__ == '__main__':
  tf.app.run()


