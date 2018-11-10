"""
Predict the column's classes by matching cells to entities
and counting the percentage of cells that are matched to a class
"""
import os
import sys
import argparse
import random
import time
from util_t2d import read_t2d_cells
from util_kb import lookup_dbo_classes

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cells_num',
    type=int,
    default=30,
    help='Number of cells in lookup')
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
    default=500,
    help='end index')
FLAGS, unparsed = parser.parse_known_args()

print 'Step #1: reading cells'
col_cells = read_t2d_cells()

print 'Step #2: lookup-based prediction column by column'
col_class_p = dict()
for col_i, col in enumerate(col_cells.keys()):
    if col_i < FLAGS.start_index:
        continue
    if col_i >= FLAGS.end_index:
        print('     This part is fully done')
        break

    cells = col_cells.get(col)
    #if len(cells) > FLAGS.cells_num:
    #    cells = random.sample(cells, FLAGS.cells_num)

    cell_classes = dict()
    unq_clses = set()
    for cell in cells:
        classes = lookup_dbo_classes(cell)
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

    if col_i % 10 == 0:
        print('     column %d annotated' % col_i)
    if (col_i + 1) % 30 == 0:
        time.sleep(60*5)


print 'Step #3: saving lookup-based predictions'
out_filename = 'p_lookup.csv'
with open(os.path.join(FLAGS.io_dir, 'predictions', out_filename), 'a') as f:
    for col_class in col_class_p.keys():
        f.write('%s,"%.2f"\n' % (col_class, col_class_p[col_class]))
