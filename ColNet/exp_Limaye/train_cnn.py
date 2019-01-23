import sys
import os
import random
import argparse
import datetime
import numpy as np
import tensorflow as tf
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from util_cnn import SyntheticColumnCNN
from util_cnn import generate_synthetic_columns
from util_cnn import synthetic_columns2sequence
from util_cnn import sequence2matrix


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir',
    type=str,
    default='~/w2v_model/enwiki_model/',
    help='Directory of word2vec model')
parser.add_argument(
    '--io_dir',
    type=str,
    default=os.path.join(current_path, 'in_out'),
    help='Directory of input/output')
parser.add_argument(
    '--cnn_dir',
    type=str,
    default=os.path.join(current_path, 'in_out/cnn'),
    help='Directory of trained models')

parser.add_argument(
    '--gap',
    type=float,
    default=1,
    help='The knowledge gap (percentage of particular entities used for training)')
parser.add_argument(
    '--train_type',
    type=int,
    default=2,
    help='0: Train CNN without fine tuning, using particular samples;'
         '1: Train CNN without fine tuning, using general samples;'
         '2: Train CNN without fine tuning, using general + particular samples'
         '3: Train CNN with fine tuning, training with particular samples, fine-tuning with general samples')
parser.add_argument(
    '--synthetic_column_size',
    type=int,
    default=1,
    help='Size of synthetic column')
parser.add_argument(
    '--sequence_size',
    type=int,
    default=15,
    help='Length of word sequence of synthetic column')

parser.add_argument(
    '--num_epochs',
    type=int,
    default=10,
    help='number of epochs')
parser.add_argument(
    '--dropout_keep_prob',
    type=float,
    default=0.5,
    help='dropout_keep_prob')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='batch size')
parser.add_argument(
    '--checkpoint_every',
    type=int,
    default=500,
    help='Save model after this many steps')
parser.add_argument(
    '--evaluate_every',
    type=int,
    default=500,
    help='Evaluate model on dev set after this many steps')
parser.add_argument(
    '--dev_sample_percentage',
    type=float,
    default=0.0,
    help='percentage of samples for development')
FLAGS, unparsed = parser.parse_known_args()

if not os.path.exists(FLAGS.cnn_dir):
    os.mkdir(FLAGS.cnn_dir)
cnn_dir = os.path.join(FLAGS.cnn_dir, 'cnn_%d_%d_%.2f' % (FLAGS.synthetic_column_size, FLAGS.train_type, FLAGS.gap))
if not os.path.exists(cnn_dir):
    os.mkdir(cnn_dir)


def read_cls_entities(file_name):
    cls_entities = dict()
    with open(os.path.join(FLAGS.io_dir, file_name), 'r') as fun_f:
        for line in fun_f.readlines():
            line_tmp = line.strip().split('","')
            line_tmp[0] = line_tmp[0][1:]
            line_tmp[-1] = line_tmp[-1][:-1]
            cls_entities[line_tmp[0]] = line_tmp[1:]
    return cls_entities


def align_samples(pos, neg):
    if len(pos) <= len(neg):
        pos_new = pos * (len(neg) / len(pos))
        neg_new = neg * 1
        pos_new += random.sample(pos, len(neg_new) - len(pos_new))
    else:
        neg_new = neg * (len(pos) / len(neg))
        pos_new = pos * 1
        neg_new += random.sample(neg, len(pos_new) - len(neg_new))
    return pos_new, neg_new


def batch_iter(data, shuffle=True):
    """
        generate batches of data
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / FLAGS.batch_size) + 1
    for epoch in range(FLAGS.num_epochs):
        if shuffle:
            batch_shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[batch_shuffle_indices]
        else:
            shuffled_data = data
        if num_batches_per_epoch > 0:
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * FLAGS.batch_size
                end_index = min((batch_num + 1) * FLAGS.batch_size, data_size)
                yield shuffled_data[start_index:end_index]
        else:
            yield shuffled_data


def train(x_train, y_train, x_dev, y_dev, cls_name, x_train_ft=None, y_train_ft=None):
    """
        Train the CNN model
    """
    cls_dir = os.path.join(cnn_dir, cls_name)

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = SyntheticColumnCNN(
                sequence_length=FLAGS.sequence_size,
                num_classes=y_train.shape[1],
                embedding_size=w2v_model.vector_size,
                channel_num=1,
                filter_sizes=[2, 3, 4],
                num_filters=3)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(cls_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(cls_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(cls_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(train_x_batch, train_y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: train_x_batch,
                    cnn.input_y: train_y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % FLAGS.evaluate_every == 0:
                    print("     {}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(dev_x_batch, dev_y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: dev_x_batch,
                    cnn.input_y: dev_y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("     {}: step {}, dev loss {:g}, dev acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            batches = batch_iter(list(zip(x_train, y_train)))

            current_step = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0 and x_dev.shape[0] > 0:
                    print("\n       Evaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                # if current_step % FLAGS.checkpoint_every == 0:
                #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #    print("     Saved model checkpoint to {}\n".format(os.path.basename(path)))

            if x_train_ft is not None and x_train_ft.shape[0] > 0 \
                    and y_train_ft is not None and y_train_ft.shape[0] > 0:

                batches_ft = batch_iter(list(zip(x_train_ft, y_train_ft)))
                for batch_ft in batches_ft:
                    x_batch_ft, y_batch_ft = zip(*batch_ft)
                    train_step(x_batch_ft, y_batch_ft)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0 and x_dev.shape[0] > 0:
                        print("\n       Evaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    # if current_step % FLAGS.checkpoint_every == 0:
                    #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #    print("     Saved [Fine-tuned] model checkpoint to {}\n".format(os.path.basename(path)))

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("     Saved model checkpoint to {}\n".format(os.path.basename(path)))


def embedding(entities_positive, entities_negative):
    # embedding
    units_positive = generate_synthetic_columns(entities_positive, FLAGS.synthetic_column_size)
    units_negative = generate_synthetic_columns(entities_negative, FLAGS.synthetic_column_size)

    sequences_positive = list()
    for ent_unit in units_positive:
        sequences_positive.append(synthetic_columns2sequence(ent_unit, FLAGS.sequence_size))
    sequences_negative = list()
    for ent_unit in units_negative:
        sequences_negative.append(synthetic_columns2sequence(ent_unit, FLAGS.sequence_size))

    x = np.zeros((len(sequences_positive) + len(sequences_negative), FLAGS.sequence_size, w2v_model.vector_size, 1))
    for sample_i, sequence in enumerate(sequences_positive + sequences_negative):
        x[sample_i] = sequence2matrix(sequence, FLAGS.sequence_size, w2v_model)

    y_positive = np.zeros((len(sequences_positive), 2))
    y_positive[:, 1] = 1.0
    y_negative = np.zeros((len(sequences_negative), 2))
    y_negative[:, 0] = 1.0
    y = np.concatenate((y_positive, y_negative))

    # shuffling
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(y.shape[0]))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    return x_shuffled, y_shuffled


if FLAGS.train_type == 0:
    print('Train CNN without fine tuning, using particular samples')
elif FLAGS.train_type == 1:
    print('Train CNN without fine tuning, using general samples')
elif FLAGS.train_type == 2:
    print('Train CNN without fine tuning, using general + particular samples')
elif FLAGS.train_type == 3:
    print('Train CNN with fine tuning, training with particular samples, fine-tuning with general samples')
else:
    sys.exit(0)


print 'Step #1: read classes'
classes = set()
with open(os.path.join(FLAGS.io_dir, 'particular_pos_samples.csv'), 'r') as f:
    for l in f.readlines():
        l_tmp = l.strip().split('","')
        classes.add(l_tmp[0][1:])

print 'Step #2: read positive and negative particular entities, positive general entities'
cls_pos_par_entities = read_cls_entities('particular_pos_samples.csv')
cls_neg_par_entities = read_cls_entities('particular_neg_samples.csv')
cls_pos_gen_entities = read_cls_entities('general_pos_samples.csv')

print('Step #3: load word2vec model')
#w2v_model = Word2Vec.load(os.path.join(FLAGS.model_dir, 'word2vec_gensim'))
w2v_model = KeyedVectors.load_word2vec_format(os.path.join(FLAGS.model_dir, 'word2vec_gensim.bin'), binary=True)

print('Step #4: train class by class')
for cls in classes:
    print('\nclass: %s' % cls)
    print('     %d general pos entities; %d particular pos entities; %d particular neg entities' %
          (len(cls_pos_gen_entities[cls]), len(cls_pos_par_entities[cls]), len(cls_neg_par_entities[cls])))

    # without fine tuning
    if FLAGS.train_type < 2:
        entities_neg = cls_neg_par_entities[cls]
        if FLAGS.train_type == 0:
            entities_pos = cls_pos_par_entities[cls]
        elif FLAGS.train_type == 1:
            entities_pos = cls_pos_gen_entities[cls]
        else:
            entities_pos = cls_pos_par_entities[cls] + cls_pos_gen_entities[cls]

        p_ents, n_ents = align_samples(entities_pos, entities_neg)

        X, Y = embedding(p_ents, n_ents)
        dev_sample_index = int(FLAGS.dev_sample_percentage * float(X.shape[0]))
        X_train, X_dev = X[dev_sample_index:], X[:dev_sample_index]
        Y_train, Y_dev = Y[dev_sample_index:], Y[:dev_sample_index]
        print('     train size: %d, dev size: %d' % (X_train.shape[0], X_dev.shape[0]))
        train(X_train, Y_train, X_dev, Y_dev, cls)

    # with fine tuning
    else:
        entities_neg = cls_neg_par_entities[cls]
        p_ents, n_ents = align_samples(cls_pos_gen_entities[cls], entities_neg)
        X, Y = embedding(p_ents, n_ents)
        p_ents_ft, n_ents_ft = align_samples(cls_pos_par_entities[cls], entities_neg)
        X_ft, Y_ft = embedding(p_ents_ft, n_ents_ft)

        dev_sample_index = int(FLAGS.dev_sample_percentage * float(X_ft.shape[0]))
        X_ft_train, X_dev = X_ft[dev_sample_index:], X_ft[:dev_sample_index]
        Y_ft_train, Y_dev = Y_ft[dev_sample_index:], Y_ft[:dev_sample_index]
        print('     [particular samples] train size: %d, dev size: %d' % (X_ft_train.shape[0], X_dev.shape[0]))

        train(X, Y, X_dev, Y_dev, cls, X_ft_train, Y_ft_train)
