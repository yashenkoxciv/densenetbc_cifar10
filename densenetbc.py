import numpy as np
import tensorflow as tf

N1, N2, N3, N4 = 6, 12, 24, 16
K = 12
THETA = 0.5

def dense_subblock(x, i, d, reuse, training, name):
    btb = tf.layers.batch_normalization(x, 3, training=training, trainable=True, name='{2}_d{0}btb{1}'.format(d, i, name), reuse=reuse)
    bta = tf.nn.leaky_relu(btb)
    btc = tf.layers.conv2d( 
        bta, 4*K, 1, 1,
        padding='same', use_bias=False,
        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / (4*K))),
        name='{2}_d{0}btc{1}'.format(d, i, name), reuse=reuse)
    
    b = tf.layers.batch_normalization(btc, 3, training=training, trainable=True, name='{2}_d{0}b{1}'.format(d, i, name), reuse=reuse)
    a = tf.nn.leaky_relu(b)
    c = tf.layers.conv2d(
        a, K, 3, 1,
        padding='same', use_bias=False,
        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / K)),
        name='{2}_d{0}c{1}'.format(d, i, name), reuse=reuse)
    nx = tf.concat((c, x), 3, name='{2}_d{0}m{1}'.format(d, i, name))
    return nx

def transpose_dense_subblock(x, i, d, reuse, training, name):
    btb = tf.layers.batch_normalization(x, 3, training=training, trainable=True, name='{2}_d{0}btb{1}'.format(d, i, name), reuse=reuse)
    bta = tf.nn.leaky_relu(btb)
    btc = tf.layers.conv2d_transpose( 
        bta, 4*K, 1, 1,
        padding='same', use_bias=False,
        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / (4*K))),
        name='{2}_d{0}btc{1}'.format(d, i, name), reuse=reuse)
    
    b = tf.layers.batch_normalization(btc, 3, training=training, trainable=True, name='{2}_d{0}b{1}'.format(d, i, name), reuse=reuse)
    a = tf.nn.leaky_relu(b)
    c = tf.layers.conv2d_transpose(
        a, K, 3, 1,
        padding='same', use_bias=False,
        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / K)),
        name='{2}_d{0}c{1}'.format(d, i, name), reuse=reuse)
    nx = tf.concat((c, x), 3, name='{2}_d{0}m{1}'.format(d, i, name))
    return nx

def dense_block(x, n, d, reuse, training, name='enc'):
    nx = x
    for i in range(n):
        nx = dense_subblock(nx, i, d, reuse, training, name)
    return nx

def transpose_dense_block(x, n, d, reuse, training, name='enc'):
    nx = x
    for i in range(n):
        nx = transpose_dense_subblock(nx, i, d, reuse, training, name)
    return nx

def encoder(x, training):
    if hasattr(encoder, 'reuse'):
        encoder.reuse = True
    else:
        encoder.reuse = False
    c1 = tf.layers.conv2d(
            x, 16, 3, 1,
            padding='same', use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 16)),
            name='enc_c1', reuse=encoder.reuse)
    nx = dense_block(c1, N1, 1, encoder.reuse, training)
    # transition layer
    b2 = tf.layers.batch_normalization(nx, 3, training=training, trainable=True, name='enc_b2', reuse=encoder.reuse)
    a2 = tf.nn.leaky_relu(b2)
    c2 = tf.layers.conv2d(
            a2, int(a2.get_shape().as_list()[3]*THETA), 1, 1,
            padding='same', use_bias=False, activation=tf.nn.leaky_relu,
            name='enc_c2', reuse=encoder.reuse)
    p2 = tf.layers.average_pooling2d(c2, 2, 2)
    
    nx = dense_block(p2, N2, 2, encoder.reuse, training)
    # transition layer
    b3 = tf.layers.batch_normalization(nx, 3, training=training, trainable=True, name='enc_b3', reuse=encoder.reuse)
    a3 = tf.nn.leaky_relu(b3)
    c3 = tf.layers.conv2d(
            a3, int(a3.get_shape().as_list()[3]*THETA), 1, 1,
            padding='same', use_bias=False, activation=tf.nn.leaky_relu,
            name='enc_c3', reuse=encoder.reuse)
    p3 = tf.layers.average_pooling2d(c3, 2, 2)
    
    nx = dense_block(p3, N3, 3, encoder.reuse, training)
    # transition layer
    b4 = tf.layers.batch_normalization(nx, 3, training=training, trainable=True, name='enc_b4', reuse=encoder.reuse)
    a4 = tf.nn.leaky_relu(b4)
    c4 = tf.layers.conv2d(
            a4, int(a4.get_shape().as_list()[3]*THETA), 1, 1,
            padding='same', use_bias=False, activation=tf.nn.leaky_relu,
            name='enc_c4', reuse=encoder.reuse)
    p4 = tf.layers.average_pooling2d(c4, 2, 2)
    
    nx = dense_block(p4, N4, 4, encoder.reuse, training)
    # ending
    b5 = tf.layers.batch_normalization(nx, 3, training=training, trainable=True, name='enc_b5', reuse=encoder.reuse)
    a5 = tf.nn.leaky_relu(b5)
    p5 = tf.reduce_mean(a5, (1, 2))
    yh_logits = tf.layers.dense(p5, 10, None, name='enc_yh', reuse=encoder.reuse)
    return yh_logits, c2, c3, c4

def dataset_accuracy(xs, ys, batch_size):
    acc = []
    for i in range(0, xs.shape[0], batch_size):
        acc_batch = sess.run(test_correct_prediction, {
                x: xs[i:i+batch_size],
                y: ys[i:i+batch_size]
        }).astype(np.float)
        acc.append(acc_batch)
    acc = np.concatenate(acc)
    return np.argwhere(acc == 1).shape[0] / xs.shape[0]

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='input_images')
    y = tf.placeholder(tf.float32, (None, 10), name='input_labels')
    vlr = tf.placeholder(tf.float32, (), name='learning_rate')
    
    yh_logits = encoder(x, training=True)
    
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y, logits=yh_logits
            )
    )
    wd = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if 'kernel' in v.name])*1e-4
    obj = loss + wd
    
    # optimizing
    lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
    set_lr = tf.assign(lr, vlr)
    
    #opt = tf.train.RMSPropOptimizer(learning_rate=0.001) #.minimize(loss)
    opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(obj, global_step=tf.train.get_global_step())
    
    # testing
    test_yh_logits = encoder(x, training=False)
    test_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y, logits=test_yh_logits
            )
    )
    
    test_correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(test_yh_logits), 1), tf.argmax(y, 1))
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
    
    # summary
    tf.summary.scalar('loss', test_loss)
    tf.summary.scalar('accuracy', test_accuracy)
    tf.summary.scalar('lr', lr)
    all_summary = tf.summary.merge_all()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer_train = tf.summary.FileWriter('summary/train', sess.graph)
    writer_test = tf.summary.FileWriter('summary/test')
    ema = [v for v in tf.global_variables() if 'moving' in v.name and 'momentum' not in v.name]
    saver = tf.train.Saver(tf.trainable_variables() + ema)
    
    # data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
    x_train = ((x_train - 128.0) / 128.0).reshape((-1, 32, 32, 3))
    x_test = ((x_test - 128.0) / 128.0).reshape((-1, 32, 32, 3))
    
    
    batch_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.15, height_shift_range=0.15,
            horizontal_flip=True
    ) # rotation_range=10, channel_shift_range=0.3, rotation_range=10
    
    def next_batch(xs, ys, batch_size):
        idx = np.random.randint(0, xs.shape[0], batch_size)
        x_batch = xs[idx]
        y_batch = ys[idx]
        return x_batch, y_batch
    
    epochs = 300
    batch_size = 64
    batches = x_train.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    summary_step = 0
    
    test_acc_th = 0.9400
    aug_batch = batch_gen.flow(x_train, y_train, batch_size=batch_size)
    #print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            if batch % 2 == 0:
                x_batch, y_batch = next_batch(x_train, y_train, batch_size)
            else:
                x_batch, y_batch = next(aug_batch)
            
            sess.run(train_op, {x: x_batch, y: y_batch})
            
            batch_step += 1
            print('\repoch {0:3.0f} {1:3.0f} %'.format(
                    epoch, batch / batches * 100
                    ), end='', flush=True
            )
            if epoch == 150:
                sess.run(set_lr, {vlr: 0.01})
            if epoch == 225:
                sess.run(set_lr, {vlr: 0.001})
            
            if batch_step % 300 == 0:
                x_batch, y_batch = next_batch(x_train, y_train, batch_size)
                train_summary_str = sess.run(all_summary, {
                    x: x_batch, y: y_batch
                })
                writer_train.add_summary(train_summary_str, batch_step)
                writer_train.flush()
                
                x_batch, y_batch = next_batch(x_test, y_test, batch_size)
                
                test_summary_str = sess.run(all_summary, {
                    x: x_batch, y: y_batch
                })
                writer_test.add_summary(test_summary_str, batch_step)
                writer_test.flush()
                summary_step += 1
        testset_acc = dataset_accuracy(x_test, y_test, 100)
        print('\repoch {0:3.0f} train_acc: {1:1.4f} test_acc: {2:1.4f}'.format(
                epoch,
                dataset_accuracy(x_train, y_train, 100),
                testset_acc
        ), end='\n', flush=True)
        if testset_acc > test_acc_th:
            saver.save(sess, 'model_{0:4.0f}/model.ckpt'.format(testset_acc*10000))
            saver.export_meta_graph(
                    'model_{0:4.0f}/model.meta'.format(testset_acc*10000),
                    collection_list=[tf.GraphKeys.UPDATE_OPS, tf.GraphKeys.TRAINABLE_VARIABLES])
            test_acc_th = testset_acc
    
    print('\rDone', ' '*25, flush=True)
    
    #saver.save(sess, 'model/model.ckpt')
    writer_train.close()
    writer_test.close()
    sess.close()
    ''' #'''

    
    
















