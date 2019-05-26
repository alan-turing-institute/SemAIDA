"""
This file is to cache the classes and super classes of given entities
"""
import os
import json
from util.util_kb import query_complete_classes_of_entity

# file to save
cache_file = 'cache_classes.json'

# read the input entities
ents = list()
cache_ents = json.load(open('cache_ents_T2D_Limaye.json'))
for v in cache_ents.values():
    ents += v
cache_ents = json.load(open('cache_ents_Wikipedia.json'))
for v in cache_ents.values():
    ents += v
ents = set(ents)

# load the existing cache
# this program can support incremental caching
ent_classes = json.load(open(cache_file)) if os.path.exists(cache_file) else dict()

print('%d left' % (len(ents) - len(ent_classes.keys())))
for i, ent in enumerate(ents):
    if ent not in ent_classes:
        classes = query_complete_classes_of_entity(ent)
        ent_classes[ent] = list(classes)
        if i % 50 == 0:
            print('i: %d done' % i)
            json.dump(ent_classes, open(cache_file, 'w'))

json.dump(ent_classes, open(cache_file, 'w'))
print('all done')
