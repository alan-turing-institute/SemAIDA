"""
Generate labeled training and testing samples (micro tables) from tables
Each dbo class has one set of columns, each column has one set of micro tables with size of M * (N + 1),
M: number of rows, N: number of surrounding columns

input:
    --io_dir, (T2D_IO, Limaye_IO, Wikipedia_IO)
    --test_name, (T2D, Limaye, Wikipedia)
    --sample_file_name, (train_cols.json, test_cols_t2d.json, test_cols_limaye.json, test_cols_wikipedia.json)

output:
    --out_dir_name, (train_samples, test_samples_t2d, test_samples_limaye, test_samples_wikipedia)

For each sample_file_name, run once
"""
import os
import sys
import json
import shutil
import argparse
from util.util_t2d import read_t2d_columns as T2D_columns
from util.util_limaye import read_limaye_columns as Limaye_columns
from util.util_wikipedia import read_wikipedia_columns as Wikipedia_columns
from util.util_micro_table import extract_samples_by_col

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
parser.add_argument('--test_name', type=str, default='Wikipedia', help='T2D or Limaye or Wikipedia')
parser.add_argument('--sample_file_name', type=str, default='test_cols_wikipedia.json',
                    help='train_cols.json or test_cols_t2d.json or test_cols_limaye.json or test_cols_wikipedia.json')
parser.add_argument('--out_dir_name', type=str, default='test_samples_wikipedia',
                    help='train_samples or test_samples_t2d or test_samples_limaye or test_samples_wikipedia')
parser.add_argument('--micro_table_size', type=str, default='5,4', help='row#,surrounding_col#')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.micro_table_size = [int(i) for i in FLAGS.micro_table_size.split(',')]

out_dir = os.path.join(FLAGS.io_dir, FLAGS.out_dir_name)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

'''Step #1: read classes, and columns of each class'''
cls_cols = json.load(open(os.path.join(FLAGS.io_dir, FLAGS.sample_file_name)))
print('%d classes' % len(cls_cols.keys()))

'''Step #2: read samples for each class'''
for cls in cls_cols:
    samples = dict()
    for col in cls_cols[cls]:
        tab_name = col.split(' ')[0]
        if FLAGS.test_name == 'T2D':
            columns = T2D_columns(tab_name)
        elif FLAGS.test_name == 'Limaye':
            columns = Limaye_columns(tab_name)
        elif FLAGS.test_name == 'Wikipedia':
            columns = Wikipedia_columns(tab_name)
        else:
            print '%s not implemented' % FLAGS.io_dir
            sys.exit(0)
        col_samples = extract_samples_by_col(columns, col, FLAGS.micro_table_size)
        samples[col] = col_samples

    with open(os.path.join(out_dir, '%s.json' % cls), 'w') as out_f:
        json.dump(samples, out_f)

    print('"%s" done' % cls)
