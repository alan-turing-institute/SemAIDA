# extract column class annotation ground truth for T2D tables
import os
import sys
import argparse
from util_kb import query_super_classes

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument(
    '--io_dir',
    type=str,
    default=os.path.join(current_path, 'in_out'),
    help='Directory of input/output')
FLAGS, unparsed = parser.parse_known_args()


print('Step #1: read fine-grained column classes')
col_fgs = dict()
with open(os.path.join(FLAGS.io_dir, 'column_gt_fg.csv'), 'r') as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        col = line_tmp[0]
        fgs = line_tmp[1:]
        col_fgs[col] = fgs

print('Step #2: read super classes')
fg_supers_cache = dict()
col_supers = dict()
for col in col_fgs.keys():
    supers = set()
    for fg in col_fgs[col]:
        tmp_supers = fg_supers_cache[fg] if fg in fg_supers_cache.keys() else query_super_classes(fg)
        supers = supers | tmp_supers
    col_supers[col] = supers


print('Step #3: write')
with open(os.path.join(FLAGS.io_dir, 'column_gt_extend_fg.csv'), 'w') as f:
    for col in col_supers.keys():
        classes_str = '"' + '","'.join(col_supers[col]) + '"'
        f.write('"%s",%s\n' % (col, classes_str))

