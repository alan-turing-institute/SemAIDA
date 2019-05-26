"""
Split columns into a training set and a testing set
"""

import csv
import random
import json


def read_class_cols(gt_file):
    cls_cols = dict()
    with open(gt_file) as f:
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            col, cls = row[0], row[1]
            if '.csv ' in col:
                col = col.replace('.csv ', ' ')
            if cls in cls_cols:
                cls_cols[cls].append(col)
            else:
                cls_cols[cls] = [col]
    return cls_cols


test_col_rate = 0.3
class_columns = read_class_cols('../T2D/col_cls.csv')

''' 
This part is to split class of T2Dv2: training and testing
get the columns of each training/testing class
'''
train_class_columns, test_class_columns = dict(), dict()
classes_num = 0
for c in class_columns:
    columns = class_columns[c]
    if len(columns) >= 2:
        test_col_num = len(columns) * test_col_rate
        test_columns = random.sample(columns, int(test_col_num) + 1)
        train_columns = [l for l in columns if l not in test_columns]
        train_class_columns[c], test_class_columns[c] = train_columns, test_columns
        print ('%s, %d/%d in T2D' % (c, len(test_columns), len(train_columns)))
        classes_num += 1
print('\nclasses # %d in T2D' % classes_num)

with open('train_cols2.json', 'w') as out_f:
    json.dump(train_class_columns, out_f)
with open('test_cols2.json', 'w') as out_f:
    json.dump(test_class_columns, out_f)

'''
This part is to get testing classes of Wikipedia and each class' column
'''
train_cols = json.load(open('train_cols.json'))
train_classes = train_cols.keys()
wikipedia_class_columns = read_class_cols('../Wikipedia/select_col_cls_GS.csv')
wikipedia_class_columns_selected = dict()
col_num = 0
for cls in wikipedia_class_columns:
    if cls in train_classes:
        columns = wikipedia_class_columns[cls]
        wikipedia_class_columns_selected[cls] = columns
        col_num += len(columns)
        print ('%s, %d in Wikipedia' % (cls, len(columns)))
print('\n testing classes #: %d, total columns #: %d in Wikipedia' % (len(wikipedia_class_columns_selected.keys()),
                                                                      col_num))
with open('test_cols_wikipedia2.json', 'w') as out_f:
    json.dump(wikipedia_class_columns_selected, out_f)

'''
This part is to get testing classes of Limaye and each class' column
'''
train_cols = json.load(open('train_cols.json'))
train_classes = train_cols.keys()
limaye_class_columns = read_class_cols('../Limaye/Limaye_col_cls.csv')
limaye_class_columns_selected = dict()
col_num = 0
for cls in limaye_class_columns:
    if cls in train_classes:
        columns = limaye_class_columns[cls]
        limaye_class_columns_selected[cls] = columns
        col_num += len(columns)
        print ('%s, %d in Limaye' % (cls, len(columns)))
print('\n testing classes #: %d, total columns #: %d in Limaye' % (len(limaye_class_columns_selected.keys()),
                                                                   col_num))
with open('test_cols_limaye2.json', 'w') as out_f:
    json.dump(limaye_class_columns_selected, out_f)
