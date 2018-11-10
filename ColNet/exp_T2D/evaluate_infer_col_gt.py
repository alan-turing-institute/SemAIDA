"""
This file contains a script that
extracts column ground truth classes of columns of T2Dv2
output: column_gt_extend_fg.csv (best class + okay classes of each column)
"""
import os
import sys
import argparse
from util_kb import super_classes
from util_t2d import read_col_gt

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument(
    '--io_dir',
    type=str,
    default=os.path.join(current_path, 'in_out'),
    help='Directory of input/output')
FLAGS, unparsed = parser.parse_known_args()

col_classes = read_col_gt()
col_classes = super_classes(col_classes)

with open(os.path.join(FLAGS.io_dir, 'column_gt_extend_fg.csv'), 'w') as f:
    for col in col_classes.keys():
        classes_str = '"' + '","'.join(col_classes[col]) + '"'
        f.write('"%s",%s\n' % (col, classes_str))

