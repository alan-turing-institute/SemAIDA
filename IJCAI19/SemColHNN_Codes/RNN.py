"""
Train and test with RNN

Input:
    --io_dir (context: training and testing split)
    --test_name (T2D-Te, Limaye, Wikipedia)

Output:
    column-level accuracy will be printed
    the trained network will be saved under io_dir/rnn
"""
import sys
import os
import json
import shutil
import argparse
import numpy as np
from gensim.models import Word2Vec
from util.util_t2d import read_t2d_columns as T2D_columns
from util.util_limaye import read_limaye_columns as Limaye_columns
from util.util_wikipedia import read_wikipedia_columns as Wikipedia_columns
from util.util_micro_table import Table_Encode_WV
from util.util_rnn import rnn_train, rnn_predict

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument('--wv_model', type=str, default='enwiki_model/')
parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
parser.add_argument('--test_name', type=str, default='T2D', help='T2D or Limaye or Wikipedia')

parser.add_argument('--micro_table_size', type=str, default='5,4', help='Must be the same as samples.py')
parser.add_argument('--cell_seq_size', type=int, default=18)
parser.add_argument('--rnn_hidden_size', type=int, default=200)
parser.add_argument('--attention_size', type=int, default=50)
parser.add_argument('--num_epochs', type=int, default=7)
parser.add_argument('--dropout_keep_prob', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--checkpoint_every', type=int, default=500, help='Save model after this many steps')
parser.add_argument('--evaluate_every', type=int, default=500, help='Evaluate model after this many steps')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.micro_table_size = [int(i) for i in FLAGS.micro_table_size.split(',')]
print FLAGS


def col_predict(mtabs):
    X_test = list()
    for mt in mtabs:
        x = Table_Encode_WV(micro_table=mt, table_size=FLAGS.micro_table_size, w2v_model=w2v_model,
                            cell_seq_size=FLAGS.cell_seq_size)
        X_test.append(x[0, 0, :, :])
    Y_test = rnn_predict(test_x=X_test, rnn_dir=rnn_dir)
    return np.average(Y_test, axis=0)


rnn_dir = os.path.join(FLAGS.io_dir, 'rnn')
if os.path.exists(rnn_dir):
    shutil.rmtree(rnn_dir)
os.mkdir(rnn_dir)


sample_dir = os.path.join(FLAGS.io_dir, 'train_samples')
if not os.path.exists(sample_dir):
    print('%s: no samples' % FLAGS.io_dir)
    sys.exit(0)
print('sampling ...')

'''get samples (micro tables) of each class'''
clses, cls_mtabs = list(), dict()
for cls_file in os.listdir(sample_dir):
    cls = cls_file.split('.')[0]
    clses.append(cls)
    col_mtabs = json.load(open(os.path.join(sample_dir, cls_file)))
    cls_mtabs[cls] = sum(col_mtabs.values(), [])

'''load word2vec model'''
w2v_model = Word2Vec.load(os.path.join(FLAGS.wv_model, 'word2vec_gensim'))

''' encode micro tables with cell_seq_size word vectors
    first cell to one sample 
        as we use sliding window for extracting micro tables
            and not consider surrounding columns'''
X, Y = list(), list()
for i, cls in enumerate(clses):
    for mtab in cls_mtabs[cls]:
        x = Table_Encode_WV(micro_table=mtab, table_size=FLAGS.micro_table_size, w2v_model=w2v_model,
                            cell_seq_size=FLAGS.cell_seq_size)
        X.append(x[0, 0, :, :])
        Y.append(np.eye(len(clses))[i])
X, Y = np.array(X), np.array(Y)
shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle_indices], Y[shuffle_indices]
print('\t train size: ' + str(X.shape))


print('training ...')
rnn_train(x_train=X, y_train=Y, FLAGS=FLAGS, w2v_model=w2v_model, rnn_dir=rnn_dir)

print('evaluating ...')
total_col_num, total_right_num, total_acc = 0, 0, 0.0

# testing columns come from T2D
if FLAGS.test_name == 'T2D':
    cls_cols = json.load(open(os.path.join(FLAGS.io_dir, 'test_cols_t2d.json')))
    test_classes_num = len(clses)
    for i, cls in enumerate(clses):     # testing classes are the same as the training classes
        sample_file = os.path.join(FLAGS.io_dir, 'test_samples_t2d', '%s.json' % cls)
        col_mtabs = json.load(open(sample_file))
        cols = cls_cols[cls]
        cls_right_num = 0
        for col in cols:
            columns = T2D_columns(col.split(' ')[0])
            p_pre = col_predict(mtabs=col_mtabs[col])

            i_pre = np.argmax(p_pre)
            if i_pre == i:
                total_right_num += 1
                cls_right_num += 1
            total_col_num += 1

        acc = (float(cls_right_num) / len(cols))
        total_acc += acc
        print('%s, %d/%d, %.4f' % (cls, cls_right_num, len(cols), acc))

# testing columns come from Limaye
elif FLAGS.test_name == 'Limaye':
    cls_cols = json.load(open(os.path.join(FLAGS.io_dir, 'test_cols_limaye.json')))
    test_classes = cls_cols.keys()
    test_classes_num = len(test_classes)
    for test_cls in test_classes:       # testing classes are a subset of the training classes
        sample_file = os.path.join(FLAGS.io_dir, 'test_samples_limaye', '%s.json' % test_cls)
        col_mtabs = json.load(open(sample_file))
        test_cols = cls_cols[test_cls]
        cls_right_num = 0
        for col in test_cols:
            columns = Limaye_columns(col.split(' ')[0])
            p_pre = col_predict(mtabs=col_mtabs[col])

            # the test class with top-1 score
            cls_top1, p_top1 = '', -1
            for cls in test_classes:
                cls_i = clses.index(cls)
                if p_pre[cls_i] >= p_top1:
                    cls_top1, p_top1 = cls, p_pre[cls_i]

            if cls_top1 == test_cls:
                total_right_num += 1
                cls_right_num += 1
            total_col_num += 1

        acc = (float(cls_right_num) / len(test_cols))
        print('%s, %d/%d, %.4f' % (test_cls, cls_right_num, len(test_cols), acc))
        total_acc += acc

# testing columns come from Wikipedia
elif FLAGS.test_name == 'Wikipedia':
    cls_cols = json.load(open(os.path.join(FLAGS.io_dir, 'test_cols_wikipedia.json')))
    test_classes = cls_cols.keys()
    test_classes_num = len(test_classes)
    for test_cls in test_classes:       # testing classes are a subset of the training classes
        sample_file = os.path.join(FLAGS.io_dir, 'test_samples_wikipedia', '%s.json' % test_cls)
        col_mtabs = json.load(open(sample_file))
        test_cols = cls_cols[test_cls]
        cls_right_num = 0
        for col in test_cols:
            columns = Wikipedia_columns(col.split(' ')[0])
            p_pre = col_predict(mtabs=col_mtabs[col])

            # the test class with top-1 score
            cls_top1, p_top1 = '', -1
            for cls in test_classes:
                cls_i = clses.index(cls)
                if p_pre[cls_i] >= p_top1:
                    cls_top1, p_top1 = cls, p_pre[cls_i]

            if cls_top1 == test_cls:
                total_right_num += 1
                cls_right_num += 1
            total_col_num += 1

        acc = (float(cls_right_num) / len(test_cols))
        print('%s, %d/%d, %.4f' % (test_cls, cls_right_num, len(test_cols), acc))
        total_acc += acc

# others
else:
    print 'test for %s not implemented' % FLAGS.test_name
    sys.exit(0)

print('average: %.4f' % (total_acc / len(clses)))
print('overall: %d/%d, %.4f' % (total_right_num, total_col_num, (float(total_right_num) / float(total_col_num))))
