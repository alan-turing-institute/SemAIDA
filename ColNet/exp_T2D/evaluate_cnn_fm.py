"""
This file is to evaluate cnns of falsely matched classes
"""
import os
import sys
import random
import argparse
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from util_t2d import read_t2d_cells
from util_cnn import ordered_cells2synthetic_columns
from util_cnn import sequence2matrix
from util_cnn import synthetic_columns2sequence

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
    '--category',
    type=str,
    default='museum',
    help='Categories of dbo_classes to evaluate')
parser.add_argument(
    '--cnn_evaluate',
    type=str,
    default=os.path.join(current_path, 'in_out/cnn/cnn_4_3_1.00'),
    help='Directory of trained models')
parser.add_argument(
    '--synthetic_column_size',
    type=int,
    default=4,
    help='Size of synthetic column')
parser.add_argument(
    '--sequence_size',
    type=int,
    default=20,
    help='Length of word sequence of entity unit')
FLAGS, unparsed = parser.parse_known_args()


def predict_unit(test_x, classifier_name):
    checkpoint_dir = os.path.join(FLAGS.cnn_evaluate, classifier_name, 'checkpoints')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
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

print 'Step #2: reading gt classes, and its columns'
class_cols = dict()
with open(os.path.join(FLAGS.io_dir, 'column_gt_extend_fg.csv'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        col = line_tmp[0]
        for cls in line_tmp[1:]:
            if cls in class_cols:
                cols_tmp = class_cols[cls]
                cols_tmp.append(col)
                class_cols[cls] = cols_tmp
            else:
                class_cols[cls] = [col]

print 'Step #3: reading FM classes and the columns, filtered by category'
fm_classes = set()
if FLAGS.category == 'all':
    for classifier in cnn_classifiers:
        if classifier not in class_cols.keys():
            fm_classes.add(classifier)
else:
    with open(os.path.join(FLAGS.io_dir, 'dbo_classes', '%s_classes.csv' % FLAGS.category), 'r') as f:
        for line in f.readlines():
            cls = line.strip()
            if cls not in class_cols.keys() and cls in cnn_classifiers:
                fm_classes.add(cls)
print('     %d FM classes' % len(fm_classes))


print 'Step #4: reading cells'
col_cells = read_t2d_cells()
all_cells = list()
for cells in col_cells.values():
    all_cells = all_cells + cells

print 'Step #5: predicting class by class'
print('     load word2vec model ...')
w2v_model = Word2Vec.load(os.path.join(FLAGS.model_dir, 'word2vec_gensim'))
fm_class_as = dict()
for fm_cls in fm_classes:
    neg_cells = random.sample(all_cells, 3000)
    neg_units = ordered_cells2synthetic_columns(neg_cells, FLAGS.synthetic_column_size)
    neg_x = np.zeros((len(neg_units), FLAGS.sequence_size, w2v_model.vector_size, 1))
    for i, unit in enumerate(neg_units):
        neg_sequence = synthetic_columns2sequence(unit, FLAGS.sequence_size)
        neg_x[i] = sequence2matrix(neg_sequence, FLAGS.sequence_size, w2v_model)
    p = predict_unit(neg_x, fm_cls)
    AS = np.average(p)
    fm_class_as[fm_cls] = AS
    print('%s: %.4f' % (fm_cls, AS))


print 'Step #6: printing FM classes and their ASs'
for fm_cls in fm_class_as.keys():
    print('     %s' % fm_cls)
for fm_cls in fm_class_as.keys():
    print('     %.4f' % (fm_class_as[fm_cls]))
