# lookup general samples <class, entity> pairs from DBPedia
import os
import sys
import argparse
from util_kb import query_general_entities

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--io_dir',
    type=str,
    default=os.path.join(current_path, 'in_out'),
    help='Directory of input/output')
FLAGS, unparsed = parser.parse_known_args()


print 'Step #1: Read candidate classes and their particular entities'
cls_par_entities = dict()
with open(os.path.join(FLAGS.io_dir, 'particular_pos_samples.csv'), 'r') as f:
    for line in f.readlines():
        line_tmp = line.strip().split('","')
        line_tmp[0] = line_tmp[0][1:]
        line_tmp[-1] = line_tmp[-1][:-1]
        cls_par_entities[line_tmp[0]] = line_tmp[1:]

print 'Step #2: Query general entities'
cls_gen_entities = query_general_entities(cls_par_entities, 1500)

print 'Step #3: Output general samples'
with open(os.path.join(FLAGS.io_dir, 'general_pos_samples.csv'), 'w') as f:
    for cls in cls_gen_entities.keys():
        entities = cls_gen_entities[cls]
        ent_s = ''
        for ent in entities:
            ent_s += ('"%s",' % ent)
        if len(ent_s) > 0:
            f.write('"%s",%s\n' % (cls, ent_s[:-1]))
        else:
            f.write('"%s"\n' % cls)
