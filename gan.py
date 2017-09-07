import numpy as np
import tensorflow as tf
from tqdm import *
import os
from utils import *


class Vaegan():
    def __init__(self, batch_size=10, cube_len=64,
                 dim_z = 200, dim_w1 = 512, dim_w2 = 256, dim_w3=128,
                 dim_w4=64, dim_w5=1):
        self.batch_size = batch_size
        self.cube_len = cube_len
        
        self.dim_z = dim_z
        self.dim_w1 = dim_w1
        self.dim_w2 = dim_w2
        self.dim_w3 = dim_w3
        self.dim_w4 = dim_w4
        self.dim_w5 = dim_w5

        self.init_weight()
        self.init_bias()
    
    def init_weight(self):
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.wg1 = tf.get_variable('gen_w1', shape=[4, 4, 4, self.dim_w1, self.dim_z], initializer=xavier_init)
        self.wg2 = tf.get_variable('gen_w2', shape=[4, 4, 4, self.dim_w2, self.dim_w1], initializer=xavier_init)
        self.wg3 = tf.get_variable('gen_w3', shape=[4, 4, 4, self.dim_w3, self.dim_w2], initializer=xavier_init)
        self.wg4 = tf.get_variable('gen_w4', shape=[4, 4, 4, self.dim_w4, self.dim_w3], initializer=xavier_init)
        self.wg5 = tf.get_variable('gen_w5', shape=[4, 4, 4, self.dim_w5, self.dim_w4], initializer=xavier_init)


        self.wd1 = tf.get_variable('dis_w1', shape=[4, 4, 4, self.dim_w5, self.dim_w4], initializer=xavier_init)
        self.wd2 = tf.get_variable('dis_w2', shape=[4, 4, 4, self.dim_w4, self.dim_w3], initializer=xavier_init)
        self.wd3 = tf.get_variable('dis_w3', shape=[4, 4, 4, self.dim_w3, self.dim_w2], initializer=xavier_init)
        self.wd4 = tf.get_variable('dis_w4', shape=[4, 4, 4, self.dim_w2, self.dim_w1], initializer=xavier_init)
        self.wd5 = tf.get_variable('dis_w5', shape=[4, 4, 4, self.dim_w1, 1], initializer=xavier_init)

        self.encw1 = tf.get_variable('enc_w1', shape=[11, 11, 3, 64], initializer=xavier_init)
        self.encw2 = tf.get_variable('enc_w2', shape=[5, 5, 64, 128], initializer=xavier_init)
        self.encw3 = tf.get_variable('enc_w3', shape=[5, 5, 128, 256], initializer=xavier_init)
        self.encw4 = tf.get_variable('enc_w4', shape=[5, 5, 256, 512], initializer=xavier_init)
        self.encw5 = tf.get_variable('enc_w5', shape=[8, 8, 512, 400], initializer=xavier_init)

    def init_bias(self):
        self.bg1 = tf.get_variable('gen_b1', shape=[512], initializer=tf.zeros_initializer())
        self.bg2 = tf.get_variable('gen_b2', shape=[256], initializer=tf.zeros_initializer())
        self.bg3 = tf.get_variable('gen_b3', shape=[128], initializer=tf.zeros_initializer())
        self.bg4 = tf.get_variable('gen_b4', shape=[64], initializer=tf.zeros_initializer())
        self.bg5 = tf.get_variable('gen_b5', shape=[1], initializer=tf.zeros_initializer())


        self.bd1 = tf.get_variable('dis_b1', shape=[64], initializer=tf.zeros_initializer())
        self.bd2 = tf.get_variable('dis_b2', shape=[128], initializer=tf.zeros_initializer())
        self.bd3 = tf.get_variable('dis_b3', shape=[256], initializer=tf.zeros_initializer())
        self.bd4 = tf.get_variable('dis_b4', shape=[512], initializer=tf.zeros_initializer())
        self.bd5 = tf.get_variable('dis_b5', shape=[1], initializer=tf.zeros_initializer())

        self.encb1 = tf.get_variable('enc_b1', initializer=np.random.rand(0, 1))
        self.encb2 = tf.get_variable('enc_b2', initializer=np.random.rand(0, 1))
        self.encb3 = tf.get_variable('enc_b3', initializer=np.random.rand(0, 1))
        self.encb4 = tf.get_variable('enc_b4', initializer=np.random.rand(0, 1))
        self.encb5 = tf.get_variable('enc_b5', initializer=np.random.rand(0, 1))

    def generate(self, z, phase=True, reuse=False):
        with tf.variable_scope('gen', reuse=reuse):
            z = tf.reshape(z, (self.batch_size, 1, 1, 1, self.dim_z))
            g_1 = tf.nn.conv3d_transpose(z, self.wg1, (self.batch_size, 4, 4, 4, self.dim_w1), strides=[1, 1, 1, 1, 1], padding='VALID')
            g_1 = tf.nn.bias_add(g_1, self.bg1)
            g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase)
            g_1 = tf.nn.relu(g_1)
            self._activation_summary(g_1)

            g_2 = tf.nn.conv3d_transpose(g_1, self.wg2, (self.batch_size, 8, 8, 8, self.dim_w2), strides=[1, 2, 2, 2, 1], padding='SAME')
            g_2 = tf.nn.bias_add(g_2, self.bg2)
            g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase)
            g_2 = tf.nn.relu(g_2)
            self._activation_summary(g_2)

            g_3 = tf.nn.conv3d_transpose(g_2, self.wg3, (self.batch_size, 16, 16, 16, self.dim_w3), strides=[1, 2, 2, 2, 1], padding='SAME')
            g_3 = tf.nn.bias_add(g_3, self.bg3)
            g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase)
            g_3 = tf.nn.relu(g_3)
            self._activation_summary(g_3)

            g_4 = tf.nn.conv3d_transpose(g_3, self.wg4, (self.batch_size, 32, 32, 32, self.dim_w4), strides=[1, 2, 2, 2, 1], padding='SAME')
            g_4 = tf.nn.bias_add(g_4, self.bg4)
            g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase)
            g_4 = tf.nn.relu(g_4)
            self._activation_summary(g_4)

            g_5 = tf.nn.conv3d_transpose(g_4, self.wg5, (self.batch_size, 64, 64, 64, self.dim_w5), strides=[1, 2, 2, 2, 1], padding='SAME')
            g_5 = tf.nn.bias_add(g_5, self.bg5)
            # Don't need batch norm
            g_5 = tf.nn.sigmoid(g_5)
            self._activation_summary(g_5)

        return g_5

    def discriminate(self, voxel, phase=True, alpha=0.2, reuse=False):
        with tf.variable_scope('dis', reuse=reuse):
            d_1 = tf.nn.conv3d(voxel, self.wd1, strides=[1, 2, 2, 2, 1], padding='SAME')
            d_1 = tf.nn.bias_add(d_1, self.bd1)
            d_1 = tf.contrib.layers.batch_norm(d_1, is_training = phase)
            d_1 = lrelu(d_1, alpha)
            self._activation_summary(d_1)

            d_2 = tf.nn.conv3d(d_1, self.wd2, strides=[1, 2, 2, 2, 1], padding='SAME')
            d_2 = tf.nn.bias_add(d_2, self.bd2)
            d_2 = tf.contrib.layers.batch_norm(d_2, is_training = phase)
            d_2 = lrelu(d_2, alpha)
            self._activation_summary(d_2)

            d_3 = tf.nn.conv3d(d_2, self.wd3, strides=[1, 2, 2, 2, 1], padding='SAME')
            d_3 = tf.nn.bias_add(d_3, self.bd3)
            d_3 = tf.contrib.layers.batch_norm(d_3, is_training = phase)
            d_3 = lrelu(d_3, alpha)
            self._activation_summary(d_3)

            d_4 = tf.nn.conv3d(d_3, self.wd4, strides=[1, 2, 2, 2, 1], padding='SAME')
            d_4 = tf.nn.bias_add(d_4, self.bd4)
            d_4 = tf.contrib.layers.batch_norm(d_4, is_training = phase)
            d_4 = lrelu(d_4, alpha)
            self._activation_summary(d_4)

            d_5 =  tf.nn.conv3d(d_4, self.wd5, strides=[1, 1, 1, 1, 1], padding='VALID')
            d_5_no_sigmoid = tf.nn.bias_add(d_5, self.bd5)
            # Don't need batch norm
            d_5 = tf.nn.sigmoid(d_5_no_sigmoid)
            self._activation_summary(d_5)

        return d_5, d_5_no_sigmoid

    def encoder(images, phase=True, reuse=False):
        with tf.variable_scope('enc', reuse=reuse):
            conv1 = tf.nn.conv2d(images, self.encw1, strides=[1, 4, 4, 1], padding='SAME')
            conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1, is_training=phase))
            conv2 = tf.nn.conv2d(conv1, self.encw2, strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2, is_training=phase))
            conv3 = tf.nn.conv2d(conv2, self.encw3, strides=[1, 2, 2, 1], padding='SAME')
            conv3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv3, is_training=phase))
            conv4 = tf.nn.conv2d(conv3, self.encw4, strides=[1, 2, 2, 1], padding='SAME')
            conv4 = tf.nn.relu(tf.contrib.layers.batch_norm(conv4, is_training=phase))
            conv5 = tf.nn.conv2d(conv4, self.encw5, strides=[1, 1, 1, 1], padding='VALID')
            conv5 = tf.nn.relu(tf.contrib.layers.batch_norm(conv5, is_training=phase))
            # conv5 is 400 dimension
            tf.reshape(conv5, [self.batch_size, 400])

            z_mean = conv5[:, :200]
            z_log_var = conv5[:, 200:]

            epsilon = tf.random_normal(tf.shape(z_log_var), name='epsilon')

            z = tf.add(z_mean, tf.sqrt(tf.exp(z_log_var)) * epsilon)
        return z

    def dis_loss(self, dis_real_logits, dis_fake_logits):
        with tf.variable_scope('dis_loss') as scope:
            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real_logits, 
                                                             labels=tf.ones_like(dis_real_logits))
            d_loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits, 
                                                              labels=tf.zeros_like(dis_fake_logits))
            d_loss = tf.reduce_mean(d_loss)
            
        return d_loss, tf.summary.scalar('discrim_loss', d_loss)


    def gen_loss(self, dis_fake):
        with tf.variable_scope('gen_loss') as scope:
            '''g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake_logits,
                                                             labels=tf.ones_like(dis_fake_logits))'''
            g_loss = tf.log(dis_fake)
            g_loss = -tf.reduce_mean(g_loss)
        return g_loss, tf.summary.scalar('generative_loss', g_loss)


    def dis_accuracy(self, dis_real, dis_fake):
        with tf.variable_scope('accuracy') as scope:
            real_correct = tf.reduce_sum(tf.cast(dis_real > 0.5, tf.int32))
            fake_correct = tf.reduce_sum(tf.cast(dis_fake < 0.5, tf.int32))
            d_acc = tf.divide(real_correct + fake_correct, 2 * self.batch_size)
        return d_acc, tf.summary.scalar('dis_accuracy', d_acc)


    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
