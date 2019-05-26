"""
Train and test with Features (FC output) learned by HNN, and prop2vec with LR and MLP
(Ensemble by a third model)

Input:
    --test_name: T2D, Limaye, Wikipedia
    --algorithm: MLP, LR
    --feature_name: the FC output of which NN model (setting)

Output:
    column-level accuracy will be printed

"""
import os
import sys
import json
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument('--wv_model', type=str, default='enwiki_model/')
parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
parser.add_argument('--test_name', type=str, default='T2D', help='T2D or Limaye or Wikipedia')
parser.add_argument('--prop2vec_dim', type=int, default=422)
parser.add_argument('--algorithm', type=str, default='MLP', help='LR or MLP')
parser.add_argument('--feature_name', type=str, default='hnnc234', help='hnnc234, hnnc23, hnnc23r23')
parser.add_argument('--micro_table_size', type=str, default='5,4')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.micro_table_size = [int(i) for i in FLAGS.micro_table_size.split(',')]
print FLAGS


def col_predict(feature_x, prop_v):
    X_test = list()
    for index, vv in enumerate(prop_v):
        vv = np.zeros(FLAGS.prop2vec_dim) if np.isnan(vv).any() else vv
        ff = feature_x[index]
        xv = np.concatenate((ff, vv))
        X_test.append(xv)
    X_test = scaler.transform(np.array(X_test))
    Y_test = clf.predict_proba(X_test)
    return np.average(Y_test, axis=0)


feature_dir = os.path.join(FLAGS.io_dir, 'train_%s_fc_%s' % (FLAGS.feature_name, FLAGS.test_name))
if not os.path.exists(feature_dir):
    print('no feature dir %s' % feature_dir)
    sys.exit(0)
prop2vec_dir = os.path.join(FLAGS.io_dir, 'train_prop2vecs')
if not os.path.exists(prop2vec_dir):
    print('no prop2vec dir %s' % prop2vec_dir)
    sys.exit(0)

print('sampling ...')

'''get samples (features and property vectors) of each train class'''
clses = list()
X, Y = list(), list()
for i, cls_file in enumerate(os.listdir(feature_dir)):
    cls = cls_file.split('.')[0]
    clses.append(cls)

    col_features = pickle.load(open(os.path.join(feature_dir, cls_file)))
    col_prop2vecs = pickle.load(open(os.path.join(prop2vec_dir, '%s.vec' % cls)))

    for col in col_features:
        features = col_features[col]
        prop2vecs = col_prop2vecs[col]
        for j, vec in enumerate(prop2vecs):
            vec = np.zeros(FLAGS.prop2vec_dim) if np.isnan(vec).any() else vec
            x = np.concatenate((features[j], vec))
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

        feature_file = os.path.join(FLAGS.io_dir, 'test_%s_T2D' % FLAGS.feature_name, '%s.vec' % cls)
        col_features = pickle.load(open(feature_file))
        vec_file = os.path.join(FLAGS.io_dir, 'test_prop2vecs_t2d', '%s.vec' % cls)
        col_prop2vecs = pickle.load(open(vec_file))

        cols = cls_cols[cls]
        cls_right_num = 0
        for col in cols:
            p_pre = col_predict(feature_x=col_features[col], prop_v=col_prop2vecs[col])

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

        feature_file = os.path.join(FLAGS.io_dir, 'test_%s_Limaye' % FLAGS.feature_name, '%s.vec' % test_cls)
        col_features = pickle.load(open(feature_file))
        vec_file = os.path.join(FLAGS.io_dir, 'test_prop2vecs_limaye', '%s.vec' % test_cls)
        col_prop2vecs = pickle.load(open(vec_file))

        test_cols = cls_cols[test_cls]
        cls_right_num = 0
        for col in test_cols:
            p_pre = col_predict(feature_x=col_features[col], prop_v=col_prop2vecs[col])

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

        feature_file = os.path.join(FLAGS.io_dir, 'test_%s_Wikipedia' % FLAGS.feature_name, '%s.vec' % test_cls)
        col_features = pickle.load(open(feature_file))
        vec_file = os.path.join(FLAGS.io_dir, 'test_prop2vecs_wikipedia', '%s.vec' % test_cls)
        col_prop2vecs = pickle.load(open(vec_file))

        test_cols = cls_cols[test_cls]
        cls_right_num = 0
        for col in test_cols:
            p_pre = col_predict(feature_x=col_features[col], prop_v=col_prop2vecs[col])

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
