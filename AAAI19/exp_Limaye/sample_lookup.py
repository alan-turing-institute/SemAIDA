# lookup candidate entities and classes
import os
import sys
import argparse
from util_limaye import read_cells_by_cols
from util_kb import lookup_resources
from util_kb import query_complete_classes_of_entity

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
if not os.path.exists(FLAGS.io_dir):
    os.mkdir(FLAGS.io_dir)


print '''Step #1: Read table columns and cells'''
cols = set()
with open(os.path.join(FLAGS.io_dir, 'column_gt_fg.csv'), 'r') as f:
    for line in f.readlines():
        cols.add(line.strip().split('","')[0][1:])
print('     columns #: %d' % len(cols))
col_cells = read_cells_by_cols(cols)


print '''Step #2: Read existing entities and classes'''
ent_file = os.path.join(FLAGS.io_dir, 'lookup_entities.csv')
cls_file = os.path.join(FLAGS.io_dir, 'lookup_classes.csv')
col_cls_file = os.path.join(FLAGS.io_dir, 'lookup_col_classes.csv')
if FLAGS.start_index == 0 and \
        (os.path.exists(ent_file) or os.path.exists(cls_file) or os.path.exists(col_cls_file)):
    print('     Error: files exist')
    sys.exit(1)

ent_cls, cls_count = dict(), dict()
if os.path.exists(ent_file):
    with open(ent_file, 'r') as out_f:
        for line in out_f.readlines():
            line_tmp = line.strip().split('","')
            line_tmp[0] = line_tmp[0][1:]
            line_tmp[-1] = line_tmp[-1][:-1]
            ent_cls[line_tmp[0]] = line_tmp[1:]
if os.path.exists(cls_file):
    with open(cls_file, 'r') as out_f:
        for line in out_f.readlines():
            line_tmp = line.strip().split('","')
            cls_count[line_tmp[0][1:]] = int(line_tmp[1][:-1])
print('     entities # %d, classes # %d' % (len(ent_cls.keys()), len(cls_count.keys())))


print '''Step #3: Lookup new entities and classes'''
cell_ents_cache = dict()
for i, col in enumerate(cols):
    if i < FLAGS.start_index:
        continue
    if i >= FLAGS.end_index:
        print('     This part is fully done, %d entities added' % len(ent_cls.keys()))
        break
    cells = col_cells[col]

    col_classes = set()
    for cell in cells:
        if cell in cell_ents_cache:
            cell_ents = cell_ents_cache[cell]
        else:
            cell_ents = lookup_resources(cell)
            cell_ents_cache[cell] = cell_ents

        for ent in cell_ents:
            if ent in ent_cls.keys():
                classes = ent_cls[ent]
            else:
                classes = query_complete_classes_of_entity(ent)
                ent_cls[ent] = classes

            for cls in classes:
                col_classes.add(cls)
                if cls not in cls_count:
                    cls_count[cls] = 1
                else:
                    cls_count[cls] += 1

    with open(col_cls_file, 'a') as f:
        if len(col_classes) == 0:
            f.write('"%s"\n' % col)
        else:
            s_cls = ''
            for c in col_classes:
                s_cls += ('"%s",' % c)
            f.write('"%s",%s\n' % (col, s_cls[:-1]))

    print ('    column %d done' % i)


print '''Step #4: Update entities and classes to files '''
with open(ent_file, 'w') as out_f:
    for ent in ent_cls.keys():
        try:
            str_cls = ''
            for c in ent_cls[ent]:
                str_cls += ('"%s",' % c)
            if len(str_cls) > 0:
                out_f.write('"%s",%s\n' % (ent, str_cls[:-1]))
            else:
                out_f.write('"%s"\n' % ent)
        except UnicodeEncodeError:
            pass
with open(cls_file, 'w') as out_f:
    for cls in cls_count.keys():
        try:
            out_f.write('"%s","%d"\n' % (cls, cls_count[cls]))
        except UnicodeEncodeError:
            pass
