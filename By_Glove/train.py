#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Pdf")

import matplotlib.pyplot as plt
import os
import time, sys
import datetime
import data_helpers
import pickle
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Define Hyperparameter
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("data_dir", "data", "Provide directory location where glove vectors are unzipped")
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
figure_name = ''
steps = []
losses = []
accuracies = []
for attr, value in sorted(FLAGS.__flags.items()):
	figure_name = figure_name + str(attr.upper()) + '=' + str(value) + '_'
	print("{}={}".format(attr.upper(), value))
print("")




# 1-4
print("Loading data...")
word_index_pickle = open(FLAGS.data_dir + '/word_index_pickle', 'rb')
pickling = pickle.load(word_index_pickle)
x = pickling['word_indices']
y = pickling['y']

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Splitting for train and dev set
x_train, x_dev = x_shuffled[:-2000], x_shuffled[-2000:-1000]
y_train, y_dev = y_shuffled[:-2000], y_shuffled[-2000:-1000]
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))




# 3
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        
        # 3-1
        cnn = TextCNN(
            sequence_length=x_train.shape[1], # 41
            num_classes=y_train.shape[1], # 5
            vocab_size=399998,
            embedding_size=FLAGS.embedding_dim, # 50
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))), # [3,4,5]
            num_filters=FLAGS.num_filters, # 128
            l2_reg_lambda = 0) # regularization term
      

        # 3-2
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss) # list of tuple ( [grad1 and var1], [grad2, var2], ... )
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) # minimize loss function
        

        # 4-1.
        sess.run(tf.global_variables_initializer())

        # 4-2.
        g = open(FLAGS.data_dir + '/glove.6B.50d_pickle', 'rb')
        pickling = pickle.load(g)
        X = pickling['embedding']
        sess.run(cnn.W.assign(X))

        # 4-3.
        # 4-3-1.
        def train_step(x_batch, y_batch):

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob # 0.5
              }

            _, step,  loss, accuracy = sess.run(
                [train_op, global_step,  cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # datetime.datetime.now() 2017-10-20 22:40:32
            # datetime.datetime.now().isoformat() == '2017-12-04'
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            
        # 4-3-2.
        def dev_step(x_batch, y_batch, writer=None):

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }

            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            accuracies.append(accuracy)
            losses.append(loss)
            steps.append(step)
            
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
         
        # 1-4
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
     

        # 4-4
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0: # 100 batch : check overfitting
                print("\nEvaluation:")
                print("")
