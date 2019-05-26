"""
train with Logistic Regression, Random Forest and Multi-layer perception
predict the score,
and ensemble the score by HNN using averaging

Input:
    --test_name: Limaye, T2D, Wikipedia
    --use_property_vector: set it to 'yes'
    --score_name: the score of which NN model (setting)
    --algorithm: LR, MLP, RF-n (n means the number of trees in RF)

Output:
    column-level accuracy will be printed

"""
import os
import sys
import json
import argparse
import pickle
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from util.util_micro_table import Table_Encode_WV_Avg

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument('--wv_model', type=str, default='enwiki_model/')
parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
parser.add_argument('--test_name', type=str, default='Limaye', help='T2D or Limaye or Wikipedia')
parser.add_argument('--use_surrounding_columns', type=str, default='yes', help='yes or no')
parser.add_argument('--use_property_vector', type=str, default='yes', help='yes or no')
parser.add_argument('--prop2vec_dim', type=int, default=422)
parser.add_argument('--algorithm', type=str, default='LR', help='LR or RF-n or MLP')
parser.add_argument('--score_name', type=str, default='hnnc234', help='hnnc234, hnnc234r23, hnnc23')
parser.add_argument('--micro_table_size', type=str, default='5,4')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.use_surrounding_columns = True if FLAGS.use_surrounding_columns == 'yes' else False
FLAGS.use_property_vector = True if FLAGS.use_property_vector == 'yes' else False
FLAGS.micro_table_size = [int(i) for i in FLAGS.micro_table_size.split(',')]
print FLAGS


def col_predict(mtabs, prop2vecs=None):
    X_test = list()
    for index, mt in enumerate(mtabs):
        xt = Table_Encode_WV_Avg(micro_table=mt, table_size=FLAGS.micro_table_size, w2v_model=w2v_model,
                                 use_surrounding_columns=FLAGS.use_surrounding_columns)
        xt = xt[0].reshape(xt[0].shape[0] * xt[0].shape[1])
        if FLAGS.use_property_vector:
            vt = np.zeros(FLAGS.prop2vec_dim) if np.isnan(prop2vecs[index]).any() else prop2vecs[index]
            xt = np.concatenate((xt, vt))
        X_test.append(xt)
    X_test = scaler.transform(np.array(X_test))
    Y_test = clf.predict_proba(X_test)
    return Y_test


sample_dir = os.path.join(FLAGS.io_dir, 'train_samples')
if not os.path.exists(sample_dir):
    print('%s: no samples' % FLAGS.io_dir)
    sys.exit(0)
prop2vec_dir = os.path.join(FLAGS.io_dir, 'train_prop2vecs')
if FLAGS.use_property_vector and not os.path.exists(prop2vec_dir):
    print('%s: no prop2vec' % FLAGS.io_dir)
    sys.exit(0)
score_dir = os.path.join(FLAGS.io_dir, 'test_%s_score_%s' % (FLAGS.score_name, FLAGS.test_name))
if not os.path.exists(score_dir):
    print('%s: no scores' % score_dir)
    sys.exit(0)

print('sampling ...')

'''get samples (micro tables and property vectors) of each train class'''
clses, cls_mtabs, cls_prop2vecs = list(), dict(), dict()
for cls_file in os.listdir(sample_dir):
    cls = cls_file.split('.')[0]
    clses.append(cls)
    col_mtabs = json.load(open(os.path.join(sample_dir, cls_file)))
    cls_mtabs[cls] = [item for sublist in col_mtabs.values() for item in sublist]
    if FLAGS.use_property_vector:
        cls_col_prop2vecs = pickle.load(open(os.path.join(prop2vec_dir, '%s.vec' % cls)))
        cls_prop2vecs[cls] = [item for sublist in cls_col_prop2vecs.values() for item in sublist]

'''load word2vec model'''
w2v_model = Word2Vec.load(os.path.join(FLAGS.wv_model, 'word2vec_gensim'))

''' encode micro tables with word vector plus averaging
    first row to one sample (as we use sliding window for extracting micro tables)'''
X, Y = list(), list()
for i, cls in enumerate(clses):
    for j, mtab in enumerate(cls_mtabs[cls]):
        x = Table_Encode_WV_Avg(micro_table=mtab, table_size=FLAGS.micro_table_size, w2v_model=w2v_model,
                                use_surrounding_columns=FLAGS.use_surrounding_columns)
        x = x[0].reshape(x[0].shape[0] * x[0].shape[1])
        if FLAGS.use_property_vector:
            v = np.zeros(FLAGS.prop2vec_dim) if np.isnan(cls_prop2vecs[cls][j]).any() else cls_prop2vecs[cls][j]
            x = np.concatenate((x, v))
        X.append(x)
        Y.append(float(i))
X, Y = np.array(X), np.array(Y)
shuffle_indices = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle_indices], Y[shuffle_indices]
print('\t train size: ' + str(X.shape))

print('training ...')
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
if FLAGS.algorithm == 'LR':  # Logistic Regression
    clf = LogisticRegression()
elif FLAGS.algorithm == 'MLP':  # Multiple Layer Perception
    clf = MLPClassifier(alpha=1)
else:  # Random Forest
    n_trees = int(FLAGS.RF_or_LR.split('-')[1])
    clf = RandomForestClassifier(n_estimators=n_trees)
clf.fit(X, Y)

Y_pre = clf.predict(X)
right_n = np.where((Y - Y_pre) == 0)[0].shape[0]
print ('\t training acc: %.3f \n' % (float(right_n) / float(Y.shape[0])))

print('evaluating ...')
total_col_num, total_right_num, total_acc = 0, 0, 0.0

# testing columns from T2D
if FLAGS.test_name == 'T2D':
    cls_cols = json.load(open(os.path.join(FLAGS.io_dir, 'test_cols_t2d.json')))
    test_classes_num = len(clses)
    for i, cls in enumerate(clses):  # testing classes are the same as the training classes
        sample_file = os.path.join(FLAGS.io_dir, 'test_samples_t2d', '%s.json' % cls)
        col_mtabs = json.load(open(sample_file))
        col_prop2vecs = None
        if FLAGS.use_property_vector:
            vec_file = os.path.join(FLAGS.io_dir, 'test_prop2vecs_t2d', '%s.vec' % cls)
            col_prop2vecs = pickle.load(open(vec_file))
        score_file = os.path.join(score_dir, '%s.vec' % cls)
        col_scores = pickle.load(open(score_file))

        cols = cls_cols[cls]
        cls_right_num = 0
        for col in cols:
            if FLAGS.use_property_vector:
                p_scores = col_predict(mtabs=col_mtabs[col], prop2vecs=col_prop2vecs[col])
            else:
                p_scores = col_predict(mtabs=col_mtabs[col])
            p_pre = np.average(p_scores, axis=0)

            score = col_scores[col]
            p_pre = (p_pre + score)/2

            i_pre = np.argmax(p_pre)
            if i_pre == i:
                total_right_num += 1
                cls_right_num += 1
            total_col_num += 1

        acc = (float(cls_right_num) / len(cols))
        print('%s, %d/%d, %.4f' % (cls, cls_right_num, len(cols), acc))
        total_acc += acc

# testing columns from Limaye
elif FLAGS.test_name == 'Limaye':
    cls_cols = json.load(open(os.path.join(FLAGS.io_dir, 'test_cols_limaye.json')))
    test_classes = cls_cols.keys()
    test_classes_num = len(test_classes)
    for test_cls in test_classes:  # testing classes are a subset of the training classes
        sample_file = os.path.join(FLAGS.io_dir, 'test_samples_limaye', '%s.json' % test_cls)
        col_mtabs = json.load(open(sample_file))
        col_prop2vecs = None
        if FLAGS.use_property_vector:
            vec_file = os.path.join(FLAGS.io_dir, 'test_prop2vecs_limaye', '%s.vec' % test_cls)
            col_prop2vecs = pickle.load(open(vec_file))
        score_file = os.path.join(score_dir, '%s.vec' % test_cls)
        col_scores = pickle.load(open(score_file))

        test_cols = cls_cols[test_cls]
        cls_right_num = 0
        for col in test_cols:
            if FLAGS.use_property_vector:
                p_scores = col_predict(mtabs=col_mtabs[col], prop2vecs=col_prop2vecs[col])
            else:
                p_scores = col_predict(mtabs=col_mtabs[col])
            p_pre = np.average(p_scores, axis=0)
            score = col_scores[col]
            p_pre = (p_pre + score)/2

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


# testing columns from Wikipedia
elif FLAGS.test_name == 'Wikipedia':
    cls_cols = json.load(open(os.path.join(FLAGS.io_dir, 'test_cols_wikipedia.json')))
    test_classes = cls_cols.keys()
    test_classes_num = len(test_classes)
    for test_cls in test_classes:  # testing classes are a subset of the training classes
        sample_file = os.path.join(FLAGS.io_dir, 'test_samples_wikipedia', '%s.json' % test_cls)
        col_mtabs = json.load(open(sample_file))
        col_prop2vecs = None
        if FLAGS.use_property_vector:
            vec_file = os.path.join(FLAGS.io_dir, 'test_prop2vecs_wikipedia', '%s.vec' % test_cls)
            col_prop2vecs = pickle.load(open(vec_file))
        score_file = os.path.join(score_dir, '%s.vec' % test_cls)
        col_scores = pickle.load(open(score_file))

        test_cols = cls_cols[test_cls]
        cls_right_num = 0
        for col in test_cols:
            if FLAGS.use_property_vector:
                p_scores = col_predict(mtabs=col_mtabs[col], prop2vecs=col_prop2vecs[col])
            else:
                p_scores = col_predict(mtabs=col_mtabs[col])
            p_pre = np.average(p_scores, axis=0)
            score = col_scores[col]
            p_pre = (p_pre + score)/2

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

print('average: %.4f' % (total_acc / test_classes_num))
print('overall: %d/%d, %.4f' % (total_right_num, total_col_num, (float(total_right_num) / float(total_col_num))))


