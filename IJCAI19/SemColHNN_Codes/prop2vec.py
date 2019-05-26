"""
This file is to generate property vectors of samples (micro tables)

Input:
    --io_dir: train-test context (T2D_IO: T2D-Tr for training, T2D-Te, Wikipedia, Limaye for testing)
    --sample_type: train_samples, test_samples_{wikipedia,t2d,limaye}
    --cache_ents_file: Cache/cache_ents_{Wikipedia,T2D_Limaye}.json
    --str_match_threshold: alpha in Algorithm 1 in Section 3.2

Output:
    --output_type: train_prop2vecs, test_prop2vecs_{wikipedia,t2d,limaye} (corresponding to the input file)

"""
import os
import sys
import json
import pickle
import argparse
import sparql
import numpy as np
from util.util_kb import query_property_object
from util.util_micro_table import Is_Number
from jellyfish import jaro_distance

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
parser.add_argument('--sample_type', type=str, default='train_samples',
                    help='train_samples, test_samples_t2d, test_samples_limaye, test_samples_wikipedia')
parser.add_argument('--out_type', type=str, default='train_prop2vecs',
                    help='train_prop2vecs, test_prop2vecs_t2d, test_prop2vecs_limaye, test_prop2vecs_wikipedia')
parser.add_argument('--cache_ents_file', type=str, default='Cache/cache_ents_T2D_Limaye.json',
                    help='cache_ents_T2D_Limaye.json, cache_ents_Wikipedia.json')
parser.add_argument('--cell2entity_top_k', type=int, default=4, help='# of entities matched for each cell')
parser.add_argument('--str_match_threshold', type=float, default=0.85)
parser.add_argument('--micro_table_size', type=str, default='5,4', help='row#,surrounding_col#')
FLAGS, unparsed = parser.parse_known_args()
FLAGS.micro_table_size = [int(i) for i in FLAGS.micro_table_size.split(',')]

sample_dir = os.path.join(FLAGS.io_dir, FLAGS.sample_type)
out_dir = os.path.join(FLAGS.io_dir, FLAGS.out_type)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
M, N = FLAGS.micro_table_size
dbp_prefix = 'http://dbpedia.org/resource/'
date_types = ['http://www.w3.org/2001/XMLSchema#date', 'http://www.w3.org/2001/XMLSchema#dateTime']


''' The function returns the equality score of an entity and a cell '''
def entity_cell_score(e, c):
    if type(e) == sparql.IRI:
        e_v = str(e)
        if dbp_prefix in e_v:
            e_v = e_v.split(dbp_prefix)[1].replace('_', ' ')
            if not Is_Number(c):
                jd = jaro_distance(unicode(e_v), c)
                if jd > FLAGS.str_match_threshold:
                    return 1.0

    elif type(e) == sparql.Literal:
        try:
            e_v = str(e)
            # object is a number: equality
            if Is_Number(e_v):
                if Is_Number(c) and float(c) == float(e_v):
                    return 1.0
            # object is date or datetime: consider the year only
            elif e.datatype in date_types:
                year = e_v.split('-')[0]
                if year in c:
                    return 1.0
            # object is text
            elif e.datatype is None:
                if not Is_Number(c):
                    jd = jaro_distance(unicode(e_v), c)
                    if jd > FLAGS.str_match_threshold:
                        return 1.0
        except UnicodeEncodeError:
            pass
    return 0.0


'''Read all the properties'''
properties_file = os.path.join(FLAGS.io_dir, 'properties.json')
properties = json.load(open(properties_file))

'''Load cached entities'''
cell_entities_file = os.path.join(current_path, FLAGS.cache_ents_file)
cell_entities = json.load(open(cell_entities_file))

'''Load finished classes'''
done_classes = set()
for done_file in os.listdir(out_dir):
    done_class = done_file.split('.')[0]
    done_classes.add(done_class)

'''For each micro table, calculate the property vector of the first row'''
for sample_file in os.listdir(sample_dir):
    class_name = sample_file.split('.')[0]
    if class_name in done_classes:
        continue
    print 'class %s ' % class_name
    col_mtabs = json.load(open(os.path.join(sample_dir, sample_file)))
    cols = col_mtabs.keys()
    col_vectors = dict()
    for k, col in enumerate(cols):
        vectors = list()
        mtabs = col_mtabs[col]
        for mtab in mtabs:
            vec = np.zeros(len(properties))

            cell_0 = mtab['col_0'][0].strip()
            if cell_0 != 'NaN':
                cell_0_ents = cell_entities[cell_0]
                if len(cell_0_ents) >= FLAGS.cell2entity_top_k:
                    cell_0_ents = cell_0_ents[0:FLAGS.cell2entity_top_k]

                for sub in cell_0_ents:
                    pro_obj_s = query_property_object(ent=sub, candidate_prop=properties)
                    for p, o in pro_obj_s:
                        p_i = properties.index(p)
                        for i in range(N):
                            cell_N_i = mtab['col_N_%d' % i][0].strip()
                            if cell_N_i != 'NaN':
                                score = entity_cell_score(o, cell_N_i)
                                vec[p_i] += score

            # this is to save storage
            if np.sum(vec) > 0:
                vectors.append(vec / FLAGS.cell2entity_top_k)
            else:
                vectors.append(np.nan)

        col_vectors[col] = vectors
        if k % 5 == 0:
            print 'col: %d/%d' % ((k+1), len(cols))

    pickle.dump(col_vectors, open(os.path.join(out_dir, '%s.vec' % class_name), 'w'))
    print 'class %s done\n' % class_name
