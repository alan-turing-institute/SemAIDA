"""
This file is to cache the entities matched by cells by lookup
Cells are saved in train_samples,test_samples_t2d,test_samples_limaye and test_samples_wikipedia
Run for each file of input cells with its corresponding cache output file
"""
import os
import json
import argparse
from util.util_kb import lookup_entities

parser = argparse.ArgumentParser()
parser.add_argument('--cache_size', type=int, default=5, help='# of entities cached for each cell')
parser.add_argument('--sample_name', type=str, default='../T2D_IO/test_samples_wikipedia',
                    help='train_samples,test_samples_t2d,test_samples_limaye,test_samples_wikipedia')
parser.add_argument('--cache_f', type=str, default='cache_ents_Wikipedia.json', help='target file to save')
parser.add_argument('--start_file_index', type=int, default=0)
parser.add_argument('--end_file_index', type=int, default=30)
FLAGS, unparsed = parser.parse_known_args()


# load the existing cache
cache = json.load(open(FLAGS.cache_f)) if os.path.exists(FLAGS.cache_f) else dict()

# --start_file_index and --end_file_index are to support incremental caching
# the cache can be restarted from the file index that the last caching stops
sample_files = os.listdir(FLAGS.sample_name)
for i, f in enumerate(sample_files):
    if i < FLAGS.start_file_index:
        continue
    if i >= FLAGS.end_file_index:
        break
    sample_f = open(os.path.join(FLAGS.sample_name, f))
    col_mtabs = json.load(sample_f)
    for mtabs in col_mtabs.values():
        for mtab in mtabs:
            cell = mtab['col_0'][0].strip()
            if cell != 'NaN':
                ents = lookup_entities(cell_text=cell, top_k=FLAGS.cache_size)
                cache[cell] = ents
    print '%d, %s done' % (i, f)

# save
with open(FLAGS.cache_f, 'w') as out_f:
    json.dump(cache, out_f)
