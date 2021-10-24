import tensorflow as tf
import numpy as np
import pdb
import os
from datetime import datetime
import tensorflow.contrib.slim as slim
import inception_v1
import resnet_v1
import keras.backend.tensorflow_backend as KTF
from create_dataset import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)


labels_nums = 5  # num_of_classes
batch_size = 64  # batch_size
resize_height = 224
resize_width = 224
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]


input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
input_w = tf.placeholder(dtype=tf.float32, shape=[None], name='w')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')
input_numbers = tf.placeholder(dtype=tf.int32, name='number')
global_step = tf.Variable(tf.constant(0))
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

def func(in_put, layer_name, is_training=True):
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        bn = tf.contrib.layers.batch_norm(inputs=in_put,
                                          decay=0.9,
                                          is_training=is_training,
                                          updates_collections=None)
    return bn

def net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums, batch_input_numbers, \
summary, writer2, step):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    weights = []
    for i in range(batch_size):
        weights.append(1)
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        record, val_loss, val_acc = sess.run([summary, loss, accuracy], feed_dict={
            input_images: val_x,
            input_labels: val_y,
            input_numbers: batch_input_numbers,
            keep_prob: 1.0,
            input_w: weights,
            is_training: False}) 
        writer2.add_summary(record, step)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc


def step_train(importance, train_op, loss, out,accuracy, probs, history, index,
               train_images_batch, train_labels_batch, train_nums_batch, train_log_step,
               val_images_batch, val_labels_batch, val_nums, val_log_step,
               snapshot_prefix, snapshot):

    global input_w

    saver = tf.train.Saver()
    max_acc = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # ckpt = tf.train.get_checkpoint_state('./models/breast/re_im/')
        # saver.restore(sess,ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="../log/no_bn_colon/res/active", graph=sess.graph)
        writer2 = tf.summary.FileWriter(logdir="../log/no_bn_colon/res/active")
        for i in range(max_steps + 1):
            batch_input_images, batch_input_labels, batch_input_numbers = sess.run(
                [train_images_batch, train_labels_batch, train_nums_batch])
            # w = tf.convert_to_tensor(weight)
            # weight = np.array(w,dtype=np.float)
            # dim = tf.size(w, name=None)
            # tensor_w = tf.reshape(w,[dim,1,1])
            # compute the mean prob of the accurate class whole samples
            w_tmp = []
            for id in batch_input_numbers:
                # print(id)
                # if need warm up
                if i > 600:
                    h = history[id]
                    prob_list = np.array(h, dtype=np.float)
                    ###variance###
                    # variance = np.var(prob_list)
                    # weight = variance + (variance * variance)/(float(len(prob_list))-1.0)
                    # weight = np.sqrt(weight) + 0.2
                    ### average predict possibility of hisotry###
                    prob_mean = np.mean(prob_list)
                    weight = prob_mean 
                #######################################
                    w_tmp.append(weight)
                else:
                    w_tmp.append(1)
            w = w_tmp
            # print(w)
            weights = np.array(w, dtype=np.float)

            _, impor, _, train_loss, prob, index_, record = sess.run([out, importance, train_op, loss, probs, index, summary], feed_dict={input_images: batch_input_images,
                                                                                              input_labels: batch_input_labels,
                                                                                              input_numbers: batch_input_numbers,
                                                                                              input_w: weights,
                                                                                              global_step : i,
                                                                                              keep_prob: 0.8, is_training: True})


            if i % train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,input_labels: batch_input_labels,input_numbers: batch_input_numbers,
input_w: weights, keep_prob: 1.0,global_step : i, is_training: False})
                writer.add_summary(record, i)
                # print ('bn/BatchNorm/beta:0', (sess.run('bn/BatchNorm/beta:0')))
                # print ('bn/BatchNorm/moving_mean:0', (sess.run('bn/BatchNorm/moving_mean:0')))
                # print ('bn/BatchNorm/moving_variance:0', (sess.run('bn/BatchNorm/moving_variance:0')))
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (
                    datetime.now(), i, train_loss, train_acc))
                # print(history)

            if i % val_log_step == 0:
                mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums, batch_input_numbers, summary, writer2, i) 
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (
                    datetime.now(), i, mean_loss, mean_acc))

            if (i % snapshot == 0 and i > 0) or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix, i))
                # saver.save(sess, snapshot_prefix, global_step=i)

            if mean_acc > max_acc and mean_acc > 0.9:
                max_acc = mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models = os.path.join(
                    path, 'best_res_{}_{:.4f}.ckpt'.format(i, max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)
            for j, [number, pro, inde] in enumerate(zip(batch_input_numbers, prob, index_)):
                measure_1 = pro[inde]
                ###############################
                # dis = pro[np.argmax(pro)] - pro[inde]
                # exp_dis = np.exp(dis)
                # measure_2 = pro[inde]/exp_dis
                history[number].append(measure_1)

        coord.request_stop()
        coord.join(threads)


def train(train_record_file, history,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_shape,
          snapshot,
          snapshot_prefix):

    [base_lr, max_steps] = train_param
    [batch_size, resize_height, resize_width, depths] = data_shape

    #
    train_nums = get_example_nums(train_record_file)

    val_nums = get_example_nums(val_record_file)
    # initialize w and history

    print('train nums:%d,val nums:%d' % (train_nums, val_nums))

    train_images, train_labels, train_numbers = read_records(
        train_record_file, resize_height, resize_width)
    train_images_batch, train_labels_batch, train_numbers_batch = get_batch_images(train_images, train_labels, train_numbers,
                                                                                   batch_size=batch_size, labels_nums=labels_nums,
                                                                                   one_hot=True, shuffle=True)
    # val
    val_images, val_labels, _ = read_records(
        val_record_file, resize_height, resize_width)
    val_images_batch, val_labels_batch,_ = get_batch_images(val_images, val_labels, _,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=False)

    bn_output = func(input_images, 'bn', is_training=True)
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        out, end_points = resnet_v1.resnet_v1_50(
            inputs=input_images, num_classes=labels_nums, is_training=is_training, global_pool=True)

    def get_active_bias(input_w):
        ## predict variance of current predict possibilities####
        probs = tf.nn.softmax(out)
        mean, variance = tf.nn.moments(probs, [len(probs.get_shape())-1])
        score = tf.reciprocal(variance, name=None)
        total_score = tf.reduce_sum(score)
        weight = tf.divide(score, total_score, name=None)
        weights = tf.multiply(weight, batch_size, name=None)
        return weights

    def get_importance_2(input_w):
        # probs = tf.nn.softmax(out)
        # label_class = tf.argmax(input_labels, 1)
        # history_t = tf.convert_to_tensor(history)
        # tensor_1 = tf.Variable([1.])
        # index = input_numbers
        # score_index = tf.gather(input_w, index)
        # score = tf.subtract(tensor_1, score_index, name=None)
        # mean,variance = tf.nn.moments(history_batch, [len(history_batch.get_shape())-1])
        # score = input_prob_mean
        score = input_w
        score = tf.reciprocal(input_w, name=None)
        total_score = tf.reduce_sum(score)
        weight = tf.divide(score, total_score, name=None)
        weights = tf.multiply(weight, batch_size, name=None)
        # weights = tf.reshape(weights, [,1,1], name=None)
        # print(weights)
        return weights

    importance = get_importance_2(input_w)
    tf.summary.histogram("weight", importance)
    probs = tf.nn.softmax(out)
    index = tf.argmax(input_labels, 1)
    tf.losses.softmax_cross_entropy(
        onehot_labels=input_labels, weights=importance, logits=out)
    loss = tf.losses.get_total_loss(add_regularization_losses=False)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', accuracy)
    learning_rate = tf.train.exponential_decay(
        base_lr, global_step, 100, 0.96, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum= 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #
    with tf.control_dependencies(update_ops):
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)
        train_op = slim.learning.create_train_op(
            total_loss=loss, optimizer=optimizer)

    step_train(importance, train_op, loss, bn_output, accuracy, probs, history, index,
               train_images_batch, train_labels_batch, train_numbers_batch, train_log_step,
               val_images_batch, val_labels_batch, val_nums, val_log_step,
               snapshot_prefix, snapshot)


if __name__ == '__main__':
    train_record_file=['../origin_size_dataset/5fold/fold_2_train.tfrecords']
    val_record_file=['../origin_size_dataset/5fold/fold_2_val.tfrecords']
    nums=get_example_nums(train_record_file)
    history = []
    nums=get_example_nums(train_record_file)+get_example_nums(val_record_file)
    for i in range(nums):
        history.append([1,1])
    train_log_step = 200
    base_lr = 0.001
    max_steps = 10000
    
    train_param = [base_lr, max_steps]

    val_log_step = 400
    snapshot = 2000
    snapshot_prefix = '../models/5fold/res/active/model_active.ckpt'
    train(train_record_file=train_record_file, history=history,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
