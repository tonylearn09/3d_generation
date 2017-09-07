from gan import Vaegan
import visdom
import load_data
import tensorflow as tf
import numpy as np
import sys
import os

FLAGS = tf.app.flags.FLAGS


def train():
    vaegan = Vaegan(batch_size=FLAGS.batch_size)
    voxels = tf.placeholder(shape=[FLAGS.batch_size, vaegan.cube_len, vaegan.cube_len,
                                   vaegan.cube_len, 1], dtype=tf.float32)
    #images = tf.placeholder(shape=[vaegan.batch_size, 256, 256, 3], shape=tf.float32)
    z = tf.placeholder(shape=[FLAGS.batch_size, vaegan.dim_z], dtype=tf.float32)

    gen_res = vaegan.generate(z, phase=True, reuse=False)
    gen_test = vaegan.generate(z, phase=False, reuse=True)
    dis_real, dis_real_no_sig = vaegan.discriminate(voxels, phase=True, reuse=False)
    dis_fake, dis_fake_no_sig = vaegan.discriminate(gen_res, phase=True, reuse=True)

    # discriminator acc
    d_acc, summary_d_acc = vaegan.dis_accuracy(dis_real, dis_fake)

    # discriminator loss
    d_loss, summary_d_loss = vaegan.dis_loss(dis_real_no_sig, dis_fake_no_sig)

    # Generator loss
    g_loss, summary_g_loss = vaegan.gen_loss(dis_fake)


    dis_vars = list(filter(lambda x: x.name.startswith('dis'), tf.trainable_variables()))
    gen_vars = list(filter(lambda x: x.name.startswith('gen'), tf.trainable_variables()))


    
    dis_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='dis')
    #print(dis_update_ops)
    with tf.control_dependencies(dis_update_ops):
        # only update the weights for the discriminator network
        optimizer_d = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5).minimize(d_loss, var_list=dis_vars)

    gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gen')
    #print(gen_update_ops)
    with tf.control_dependencies(gen_update_ops):
        # only update the weights for the generator network
        optimizer_g = tf.train.AdamOptimizer(learning_rate=0.008, beta1=0.5).minimize(g_loss, var_list=gen_vars)



    saver = tf.train.Saver()
    vis = visdom.Visdom()


    with tf.Session() as sess:
        #merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train',
                                             sess.graph)
        sess.run(tf.global_variables_initializer())
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print('restore from checkpoint {0}'.format(ckpt))


        volumes = load_data.load_data(obj_class=FLAGS.obj, is_train=True)
        print('Using ' + FLAGS.obj + ' Data')

        # Make it one channel -> [FLAGS.batch_size, cube_len, cube_len, cube_len, 1]
        volumns = volumes[...,np.newaxis].astype(np.float)
        #print(volumns.shape)
        n_iter = 0
        for epoch in range(FLAGS.n_epochs):
            for batch in iterate_minibatches(volumns, FLAGS.batch_size, shuffle=True):
                x = batch
                #hidden = np.random.uniform(0, 1, size=[FLAGS.batch_size, vaegan.dim_z]).astype(np.float32)
                hidden = np.random.normal(0, 0.33, size=[FLAGS.batch_size, vaegan.dim_z]).astype(np.float32)

                dis_summary = tf.summary.merge([summary_d_loss, summary_d_acc])
                summary_dis, dis_loss, dis_acc = sess.run([dis_summary, d_loss, d_acc], feed_dict={voxels: x, z: hidden})
                train_writer.add_summary(summary_dis, epoch)

                # Train discrimator only if acc < 0.8
                if dis_acc < 0.8:
                    sess.run([optimizer_d], feed_dict={z: hidden, voxels: x})


                print('------------Discriminator training----------------')
                print('Epoch: {0}, iteration: {1}, d_loss: {2}, d_acc: {3}'.format(epoch, n_iter, dis_loss, dis_acc))

                # Always train generator
                #sess.run([optimizer_g], feed_dict={z: hidden})
                summary_gen, gen_loss, _ = sess.run([summary_g_loss, g_loss, optimizer_g], feed_dict={z: hidden})
                train_writer.add_summary(summary_gen, epoch)
                print('------------Generator training----------------')
                print('Epoch: {0}, iteration: {1}, g_loss: {2}, d_acc: {3}'.format(epoch, n_iter, gen_loss, dis_acc))
                n_iter += 1
                

            # Generate some voxels each epoch
            #hidden_sample = np.random.uniform(0, 1, size=[FLAGS.batch_size, vaegan.dim_z]).astype(np.float32)
            hidden_sample = np.random.normal(0, 0.33, size=[FLAGS.batch_size, vaegan.dim_z]).astype(np.float32)
            g_objs = sess.run([gen_test], feed_dict={z: hidden_sample})
            '''if not os.path.exists(FLAGS.out_sample_dir):
                os.makedirs(FLAGS.out_sample_dir)
                g_objs.dump(FLAGS.out_sample_dir + '/' + str(epoch))'''
            # Save each epoch
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'))



def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]



def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 10, 'size of training batches')
    tf.app.flags.DEFINE_integer('n_epochs', 5, 'number of epochs')
    #tf.app.flags.DEFINE_integer('max_steps', 200, 'number of training iterations')
    #tf.app.flags.DEFINE_integer('eval_steps', 100, 'number of iterations to evaluate')
    #tf.app.flags.DEFINE_integer('save_steps', 100, 'number of iterations to save')
    tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints/', 'path to checkpoint directory')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')
    tf.app.flags.DEFINE_string('out_sample_dir', 'samples', 'path to directory for storing GAN output samples')
    tf.app.flags.DEFINE_boolean('restore', False, 'Whether to restore checkpoint')
    tf.app.flags.DEFINE_string('obj', 'chair', 'object to train')


    tf.app.run()

