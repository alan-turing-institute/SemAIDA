"""
The program aims to sample property vectors and assist the analysis (Results in Fig.4 in paper)
"""
import random
import pickle
import json
import numpy as np

CLASS = 'Film'
TAB_NUM = 10
ROW_NUM_PER_TAB = 4

VEC = 'test_prop2vecs_wikipedia/%s.vec' % CLASS
SAMPLE = 'test_samples_wikipedia/%s.json' % CLASS
PROP = 'properties.json'


def print_row():
    mtabs = tab_mtabs[tab]
    mtab = mtabs[vec_index]
    print('%s, %s, %s, %s, %s' % (mtab['col_0'][0], mtab['col_N_0'][0], mtab['col_N_1'][0], mtab['col_N_2'][0],
                                  mtab['col_N_3'][0]))


def print_property():
    for slot, item in enumerate(vec):
        if item > 0:
            print properties[slot]


properties = json.load(open(PROP))
tab_mtabs = json.load(open(SAMPLE))
tab_vecs = pickle.load(open(VEC))
tabs = tab_vecs.keys()

for i in range(TAB_NUM):
    tab = random.sample(tabs, 1)[0]
    vecs = tab_vecs[tab]
    for j in range(ROW_NUM_PER_TAB):
        vec_index = random.sample(range(len(vecs)), 1)[0]
        vec = vecs[vec_index]
        print '\n ------------------'
        if np.isnan(vec).any():
            print 'nan'
        else:
            print_row()
            print_property()

