import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from util_limaye import read_cells_by_cols
from util_cnn import ordered_cells2synthetic_columns
from util_cnn import sequence2matrix
from util_cnn import synthetic_columns2sequence
from util_cnn import random_cells2synthetic_columns
from util_cnn import permutation_cells2synthetic_columns


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir',
    type=str,
    default='~/w2v_model/enwiki_model/',
    help='Directory of word2vec model')
parser.add_argument(
    '--synthetic_column_size',
    type=int,
    default=1,
    help='Size of synthetic column')
parser.add_argument(
    '--sequence_size',
    type=int,
    default=15,
    help='Length of word sequence of entity unit')
parser.add_argument(
    '--synthetic_column_type',
    type=int,
    default=0,
    help='synthetic column num to sample for each column; '
         '>=1: sample a number; 0: sliding window; -1: permutation combination and voting')
parser.add_argument(
    '--cnn_evaluate',
    type=str,
    default=os.path.join(current_path, 'in_out/cnn/cnn_1_2_1.00'),
    help='Directory of trained models')
parser.add_argument(
    '--io_dir',
    type=str,
    default=os.path.join(current_path, 'in_out'),
    help='Directory of input/output')

FLAGS, unparsed = parser.parse_known_args()

print('load word2vec model ...')
w2v_model = Word2Vec.load(os.path.join(FLAGS.model_dir, 'word2vec_gensim'))


def predict(test_x, classifier_name):
    checkpoint_dir = os.path.join(FLAGS.cnn_evaluate, classifier_name, 'checkpoints')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            probabilities = graph.get_operation_by_name("output/probabilities").outputs[0]

            test_p = sess.run(probabilities, {input_x: test_x, dropout_keep_prob: 1.0})

    return test_p[:, 1]


print 'Step #1: reading cnn classifiers'
cnn_classifiers = set()
for cls_name in os.listdir(FLAGS.cnn_evaluate):
    cnn_classifiers.add(cls_name)

print 'Step #2: reading col_cells and col_lookup_classes'
cols = set()
with open(os.path.join(FLAGS.io_dir, 'column_gt_fg.csv'), 'r') as f:
    for line in f.readlines():
        cols.add(line.strip().split('","')[0][1:])
print('     columns #: %d' % len(cols))
col_cells = read_cells_by_cols(cols)

col_lookup_classes = dict()
with open(os.path.join(FLAGS.io_dir, 'lookup_col_classes.csv')) as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        if len(line_tmp) > 1:
            col = line_tmp[0][1:]
            line_tmp[-1] = line_tmp[-1][:-1]
            col_lookup_classes[col] = set(line_tmp[1:])
        else:
            col = line_tmp[0][1:-1]
            col_lookup_classes[col] = set()

print 'Step #3: predicting column by column'
col_class_p = dict()
for col_i, col in enumerate(cols):
    cells = col_cells[col]
    if FLAGS.synthetic_column_type >= 0:
        if FLAGS.synthetic_column_type > 0:
            units = random_cells2synthetic_columns(cells, FLAGS.synthetic_column_size, FLAGS.synthetic_column_type)
        else:
            units = ordered_cells2synthetic_columns(cells, FLAGS.synthetic_column_size)
    else:
        units = permutation_cells2synthetic_columns(cells)

    X = np.zeros((len(units), FLAGS.sequence_size, w2v_model.vector_size, 1))
    for i, unit in enumerate(units):
        seq = synthetic_columns2sequence(unit, FLAGS.sequence_size)
        X[i] = sequence2matrix(seq, FLAGS.sequence_size, w2v_model)

    for classifier in col_lookup_classes[col]:
        if classifier in cnn_classifiers:
            col_class = '"%s","%s"' % (col, classifier)
            p = predict(X, classifier)
            score = np.mean(p)
            col_class_p[col_class] = score

    if col_i % 5 == 0:
        print('     column %d predicted' % col_i)

print 'Step #4: saving predictions'
out_file_name = 'p_%s.csv' % os.path.basename(FLAGS.cnn_evaluate)
with open(os.path.join(FLAGS.io_dir, 'predictions', out_file_name), 'w') as f:
    for col_class in col_class_p.keys():
        f.write('%s,"%.2f"\n' % (col_class, col_class_p[col_class]))
