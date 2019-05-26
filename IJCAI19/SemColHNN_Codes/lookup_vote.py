import os
import sys
import json
import argparse

from util.util_t2d import read_t2d_columns as T2D_columns
from util.util_limaye import read_limaye_columns as Limaye_columns
from util.util_wikipedia import read_wikipedia_columns as Wikipedia_columns

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
parser.add_argument('--test_name', type=str, default='Limaye', help='T2D or Limaye or Wikipedia')
parser.add_argument('--top_k', type=int, default=5)
FLAGS, unparsed = parser.parse_known_args()

if FLAGS.test_name == 'T2D' or FLAGS.test_name == 'Limaye':
    cache_ents = json.load(open(os.path.join(current_path, 'Cache/cache_ents_T2D_Limaye.json')))
else:
    cache_ents = json.load(open(os.path.join(current_path, 'Cache/cache_ents_Wikipedia.json')))
cache_classes = json.load(open(os.path.join(current_path, 'Cache/cache_classes.json')))

print('Step #1: loading test columns ...')
train_cls_cols = json.load(open(os.path.join(FLAGS.io_dir, 'train_cols.json')))
train_classes = train_cls_cols.keys()

if FLAGS.test_name == 'T2D':
    test_file = os.path.join(FLAGS.io_dir, 'test_cols_t2d.json')
elif FLAGS.test_name == 'Limaye':
    test_file = os.path.join(FLAGS.io_dir, 'test_cols_limaye.json')
else:
    test_file = os.path.join(FLAGS.io_dir, 'test_cols_wikipedia.json')
test_cls_cols = json.load(open(test_file))
cls_col_cells = dict()
for cls in test_cls_cols:
    col_cells = list()
    for col in test_cls_cols[cls]:
        tab_name, col_id = col.split(' ')
        col_id = int(col_id)
        if FLAGS.test_name == 'T2D':
            columns = T2D_columns(tab_name)
        elif FLAGS.test_name == 'Limaye':
            columns = Limaye_columns(tab_name)
        else:
            columns = Wikipedia_columns(tab_name)
        cells = columns[col_id]
        col_cells.append(cells)
    cls_col_cells[cls] = col_cells

print('Step #2: entity and class lookup ...')
hit_col_num, test_col_num = 0, 0
for cls in test_cls_cols:
    col_cells = cls_col_cells[cls]
    for cells in col_cells:
        test_col_num += 1
        votes = dict()
        for c in train_classes:
            votes[c] = 0
        for cell in cells:
            if cell not in cache_ents:
                ents = []
                print('cell: %s has no entity in cache' % cell)
            else:
                ents = cache_ents[cell]
            classes = set()
            for ent in ents[0:FLAGS.top_k]:
                for c in cache_classes[ent]:
                    classes.add(c)
            for c in classes:
                if c in votes:
                    votes[c] += 1
        if votes[cls] >= max(votes.values()) and votes[cls] > 0:
            hit_col_num += 1

print('%s, top-k: %d, accuracy: %.3f' % (FLAGS.test_name, FLAGS.top_k, float(hit_col_num) / float(test_col_num)))
