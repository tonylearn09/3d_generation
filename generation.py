import utils
from gan import Vaegan
import visdom
import load_data
import tensorflow as tf
import numpy as np
import sys
import os

FLAGS = tf.app.flags.FLAGS

def test():
    vaegan = Vaegan(batch_size=FLAGS.batch_size)
    z_hid = tf.placeholder(shape=[FLAGS.batch_size, vaegan.dim_z], dtype=tf.float32)

    #gen_res = vaegan.generate(z_hid, phase=False, reuse=True)
    gen_res = vaegan.generate(z_hid, phase=False)
    vis = visdom.Visdom()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from checkpoint {0}'.format(ckpt))

        # Generating
        for i in range(20):
            z_sample = np.random.uniform(0, 1, size=[FLAGS.batch_size, vaegan.dim_z]).astype(np.float32)
            z_sample = np.random.normal(0, 0.33, size=[FLAGS.batch_size, vaegan.dim_z]).astype(np.float32)

            gen_objects = sess.run(gen_res, feed_dict={z_hid: z_sample})
            #print(gen_objects)
            # Choose 2 to plot
            choice = np.random.randint(0, FLAGS.batch_size, 2)
            #print(choice)
            for i in range(2):
                print(gen_objects[choice[i]].max(), gen_objects[choice[i]].min(), gen_objects[choice[i]].shape)
                if gen_objects[choice[i]].max() > 0.5:
                    utils.plot(np.squeeze(gen_objects[choice[i]] > 0.5),
                               vis, '_'.join(map(str, [i])))


def main(argv=None):
    test()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 10, 'size of training batches')
    tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints/', 'path to checkpoint directory')
    tf.app.flags.DEFINE_string('obj', 'chair', 'object to train')


    tf.app.run()
