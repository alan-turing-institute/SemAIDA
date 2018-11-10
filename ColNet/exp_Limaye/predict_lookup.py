import os
import sys
import argparse
from util_limaye import read_cells_by_cols
from util_kb import lookup_dbo_classes


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--io_dir',
    type=str,
    default=os.path.join(current_path, 'in_out'),
    help='Directory of input/output')
parser.add_argument(
    '--start_index',
    type=int,
    default=0,
    help='start index')
parser.add_argument(
    '--end_index',
    type=int,
    default=90,
    help='end index')
FLAGS, unparsed = parser.parse_known_args()

print 'Step #1: reading columns and cells'
cols = set()
with open(os.path.join(FLAGS.io_dir, 'column_gt_fg.csv'), 'r') as f:
    for line in f.readlines():
        cols.add(line.strip().split('","')[0][1:])
print('     columns #: %d' % len(cols))
col_cells = read_cells_by_cols(cols)

print 'Step #2: lookup-based prediction column by column'
out_file = os.path.join(FLAGS.io_dir, 'predictions', 'p_lookup.csv')
if os.path.exists(out_file) and FLAGS.start_index == 0:
    print('     file exists')
    sys.exit(1)

cell_classes_cache = dict()
col_class_p = dict()
for col_i, col in enumerate(col_cells.keys()):
    if col_i < FLAGS.start_index:
        continue
    if col_i >= FLAGS.end_index:
        print('     This part is fully done')
        break

    cells = col_cells.get(col)

    cell_classes = dict()
    unq_clses = set()
    for cell in cells:
        classes = cell_classes_cache[cell] if cell in cell_classes_cache else lookup_dbo_classes(cell)
        cell_classes[cell] = classes
        unq_clses = unq_clses | set(classes)

    for cls in unq_clses:
        count = 0
        for cell in cells:
            if cls in cell_classes[cell]:
                count += 1
        p = float(count) / float(len(cells))
        col_class = '"%s","%s"' % (col, cls)
        col_class_p[col_class] = p

    with open(out_file, 'a') as f:
        for col_class in col_class_p.keys():
            f.write('%s,"%.2f"\n' % (col_class, col_class_p[col_class]))

    print('     column %d annotated' % col_i)

