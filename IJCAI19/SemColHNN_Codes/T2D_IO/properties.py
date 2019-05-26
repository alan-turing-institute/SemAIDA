"""
This file is to generate candidate properties

Input:
    --train_cols.json (classes)
    --property_ratio_threshold (sigma in Section 3.2)
Output:
    --properties.json (candidate properties)
"""
import os
import sys
import json
import argparse
from util.util_kb import query_ent_num
from util.util_kb import query_property_ent_num

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--property_ratio_threshold', type=float, default=0.005)
parser.add_argument('--input_file', type=str, default='train_cols.json')
parser.add_argument('--output_file', type=str, default='properties.json')
FLAGS, unparsed = parser.parse_known_args()

cls_cols = json.load(open(FLAGS.input_file))
classes = cls_cols.keys()

semantic_properties = set()
excluded_properties = ['http://dbpedia.org/ontology/wikiPageExternalLink', 'http://dbpedia.org/ontology/abstract',
                       'http://dbpedia.org/ontology/wikiPageID', 'http://dbpedia.org/ontology/thumbnail',
                       'http://dbpedia.org/ontology/wikiPageWikiLink', 'http://dbpedia.org/ontology/wikiPageRevisionID',
                       'http://dbpedia.org/ontology/wikiPageRedirects', 'http://dbpedia.org/ontology/imdbId']

for cls in classes:
    print '\n ------- %s ------- \n' % cls
    all_ent_num = query_ent_num(cls)
    prop_ent_num = query_property_ent_num(cls)
    n = 0
    for prop in prop_ent_num:
        if prop in excluded_properties:
            continue
        ent_num = prop_ent_num[prop]
        ratio = float(ent_num) / float(all_ent_num)
        if ratio >= FLAGS.property_ratio_threshold:
            print '%s %d %.4f' % (prop, ent_num, ratio)
            semantic_properties.add(prop)
            n += 1
    print 'class: %s, entities: %d, properties: %d (%d)' % (cls, all_ent_num, n, len(prop_ent_num.keys()))

print '\n %d properties saved' % len(semantic_properties)
with open(FLAGS.output_file, 'w') as out_f:
    json.dump(list(semantic_properties), out_f)
