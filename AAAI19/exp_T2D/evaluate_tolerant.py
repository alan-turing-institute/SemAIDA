"""
Evaluate the results under tolerant model
"""
import os
import sys
import argparse
from sklearn import metrics
from util_t2d import primary_key_cols

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument(
    '--threshold',
    type=float,
    default=0.48,
    help='Threshold to determine a column class')
parser.add_argument(
    '--predictions',
    type=str,
    # default=os.path.join(current_path, 'in_out/predictions/p_lookup.csv'),
    # default=os.path.join(current_path, 'in_out/predictions/p_cnn_1_2_1.00.csv'),
    default=os.path.join(current_path, 'in_out/predictions/p_cnn_1_2_1.00_lookup.csv'),
    help='File of predictions')
parser.add_argument(
    '--ground_truths',
    type=str,
    default=os.path.join(current_path, 'in_out/column_gt_extend_fg.csv'),
    help='Ground truths')
parser.add_argument(
    '--primary_key',
    type=str,
    default='no',
    help='Whether use primary key only')
FLAGS, unparsed = parser.parse_known_args()


print 'Step #1: Read column type ground truths'
col_cls_gt = set()
with open(FLAGS.ground_truths) as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        col = line_tmp[0]
        for cls in line_tmp[1:]:
            col_cls = '%s:%s' % (col, cls)
            col_cls_gt.add(col_cls)

print 'Step #2: Read positive and negative predictions'
col_cls_pos = set()
with open(FLAGS.predictions) as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        col = line_tmp[0]
        cls = line_tmp[1]
        col_cls = '%s:%s' % (col, cls)
        score = float(line_tmp[2])
        if score >= FLAGS.threshold:
            col_cls_pos.add(col_cls)

col_cls_gt2, col_cls_pos2 = set(), set()
if FLAGS.primary_key == 'yes':
    pk_cols = primary_key_cols()
    for col_cls in col_cls_gt:
        if col_cls.split(':')[0] in pk_cols:
            col_cls_gt2.add(col_cls)
    for col_cls in col_cls_pos:
        if col_cls.split(':')[0] in pk_cols:
            col_cls_pos2.add(col_cls)
    col_cls_gt, col_cls_pos = col_cls_gt2, col_cls_pos2

print 'Step #3: Calculate metrics'

hits = len(col_cls_pos.intersection(col_cls_gt))

precision = float(hits)/float(len(col_cls_pos))
recall = float(hits)/float(len(col_cls_gt))
f1 = 2 * precision * recall / (precision + recall)


print('Precision: %.4f' % precision)
print('Recall: %.4f' % recall)
print('F1 score: %.4f' % f1)
print('%.3f' % precision)
print('%.3f' % recall)
print('%.3f' % f1)

