"""
lookup candidate entities and classes
"""
import os
import sys
import argparse
from util_t2d import read_t2d_cells
from util_kb import lookup_resources

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
    default=400,
    help='start index')
parser.add_argument(
    '--end_index',
    type=int,
    default=440,
    help='end index')
FLAGS, unparsed = parser.parse_known_args()
if not os.path.exists(FLAGS.io_dir):
    os.mkdir(FLAGS.io_dir)


print '''Step #1: Read table cells'''
col_cells = read_t2d_cells()
col_num = len(col_cells.keys())
print('     columns #: %d' % col_num)


print '''Step #2: Read existing entities and classes'''
ent_file = os.path.join(FLAGS.io_dir, 'lookup_entities.csv')
cls_file = os.path.join(FLAGS.io_dir, 'lookup_classes.csv')
entities, cls_count = set(), dict()
if os.path.exists(ent_file):
    with open(ent_file, 'r') as out_f:
        for line in out_f.readlines():
            line_tmp = line.strip().split('","')
            entities.add(line_tmp[0][1:])
if os.path.exists(cls_file):
    with open(cls_file, 'r') as out_f:
        for line in out_f.readlines():
            line_tmp = line.strip().split('","')
            cls_count[line_tmp[0][1:]] = int(line_tmp[1][:-1])
print('     entities # %d, classes # %d' % (len(entities), len(cls_count.keys())))


print '''Step #3: Lookup new entities and classes'''
ent_cls = dict()
for i, col in enumerate(col_cells.keys()):
    if i < FLAGS.start_index:
        continue
    if i >= FLAGS.end_index:
        print('     This part is fully done, %d entities added' % len(ent_cls.keys()))
        break
    cells = col_cells.get(col)
    col_classes = set()
    for cell in cells:
        ent_classes = lookup_resources(cell)
        for ent in ent_classes.keys():
            if ent not in entities:
                ent_cls[ent] = ent_classes[ent]
            for cls in ent_classes[ent]:
                col_classes.add(cls)
                if cls not in cls_count:
                    cls_count[cls] = 1
                else:
                    cls_count[cls] += 1

    with open(os.path.join(FLAGS.io_dir, 'lookup_col_classes.csv'), 'a') as f:
        if len(col_classes) == 0:
            f.write('"%s"\n' % col)
        else:
            s_cls = ''
            for c in col_classes:
                s_cls += ('"%s",' % c)
            f.write('"%s",%s\n' % (col, s_cls[:-1]))

    if i % 10 == 0:
        print ('    column %d done' % i)


print '''Step #4: Update entities and classes to files '''
with open(ent_file, 'a') as out_f:
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
with open(cls_file, 'a') as out_f:
    for cls in cls_count.keys():
        try:
            out_f.write('"%s","%d"\n' % (cls, cls_count[cls]))
        except UnicodeEncodeError:
            pass
