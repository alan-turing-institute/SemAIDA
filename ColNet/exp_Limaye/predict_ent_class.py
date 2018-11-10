import os
import sys
import argparse
from util_limaye import read_cells_by_cols

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--io_dir',
    type=str,
    default=os.path.join(current_path, 'in_out'),
    help='Directory of input/output')
FLAGS, unparsed = parser.parse_known_args()

print 'Step #1: reading columns and cells of each column'
cols = set()
with open(os.path.join(FLAGS.io_dir, 'column_gt_fg.csv'), 'r') as f:
    for line in f.readlines():
        cols.add(line.strip().split('","')[0][1:])
print('     columns #: %d' % len(cols))
col_cells = read_cells_by_cols(cols)

print 'Step #2: read classes of each table'
tab_class_num = dict()
with open(os.path.join(FLAGS.io_dir, 'column_ent_class.csv'), 'r') as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        tab = line_tmp[0]
        class_num = set()
        for item in line_tmp[1:]:
            if 'Wikidata:Q11424' not in item:
                class_num.add(item)
        tab_class_num[tab] = class_num

print 'Step #3: calculate score'
col_class_score = list()
for col in cols:
    cell_num = len(col_cells[col])
    tab = col.split(' ')[0]
    class_nums = tab_class_num[tab]
    for class_num in class_nums:
        tmp = class_num.split(':')
        cls = tmp[0]
        num = int(tmp[1])
        score = float(num)/float(cell_num)
        col_class_score.append('"%s","%s","%.2f"' % (col, cls, score))

print 'Step #4: output'
out_file = os.path.join(FLAGS.io_dir, 'predictions', 'p_ent_class.csv')
with open(out_file, 'w') as f:
    for s in col_class_score:
        f.write('%s\n' % s)
