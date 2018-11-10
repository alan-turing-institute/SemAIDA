"""
Evaluate the results under restrict model
"""
import os
import sys
import argparse
from util_t2d import primary_key_cols

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument(
    '--threshold',
    type=float,
    default=0.16,
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
    default='yes',
    help='Whether use primary key only')
FLAGS, unparsed = parser.parse_known_args()


print 'Step #1: Read column, its fine grained ground truth and all the ground truths'
col_gt_classes = dict()
col_gt_fg_cls = dict()
with open(FLAGS.ground_truths) as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        col = line_tmp[0]
        col_gt_fg_cls[col] = line_tmp[1]
        col_gt_classes[col] = set(line_tmp[1:])

print 'Step #2: Read column, its predicted classes and scores'
col_pclasses = dict()
with open(FLAGS.predictions) as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        col = line_tmp[0]
        cls = line_tmp[1]
        score = float(line_tmp[2])
        if score >= FLAGS.threshold:
            if col in col_pclasses:
                col_pclasses[col].add(cls)
            else:
                col_pclasses[col] = {cls}

print 'Step #4: Calculate metrics'
hits, p_num, gt_num = 0, 0, 0
pk_cols = primary_key_cols()
for col in col_pclasses.keys():
    pclasses = col_pclasses[col]
    if FLAGS.primary_key == 'yes':
        if col in pk_cols:
            p_num += len(pclasses)
            if len(col_gt_classes[col] - pclasses) == 0:
                hits += len(pclasses.intersection(col_gt_classes[col]))
    else:
        p_num += len(pclasses)
        if len(col_gt_classes[col] - pclasses) == 0:
            hits += len(pclasses.intersection(col_gt_classes[col]))

for col in col_gt_classes.keys():
    if FLAGS.primary_key == 'yes':
        if col in pk_cols:
            gt_num += len(col_gt_classes[col])
    else:
        gt_num += len(col_gt_classes[col])

precision = float(hits)/float(p_num)
recall = float(hits)/float(gt_num)
if precision == 0 or recall == 0:
    f1 = 0
else:
    f1 = 2 * precision * recall / (precision + recall)

print('Precision: %.4f' % precision)
print('Recall: %.4f' % recall)
print('F1 score: %.4f' % f1)
print('%.4f' % precision)
print('%.4f' % recall)
print('%.4f' % f1)


