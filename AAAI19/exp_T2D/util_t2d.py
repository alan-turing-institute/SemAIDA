"""
This file contains functions related to processing of T2Dv2 table sets
"""
import os
import sys

t2d_dir = '../T2Dv2'
if not os.path.exists(t2d_dir):
    print('%s does not exist' % t2d_dir)
    sys.exit(1)


# read columns, each of which includes table id and column number, separated by ' '
def read_table_cols():
    tab_cols = list()
    with open(os.path.join(t2d_dir, 'col_class_checked_fg.csv')) as f:
        lines = f.readlines()
        for line in lines:
            line_tmp = line.strip().split('","')
            line_tmp[0] = line_tmp[0][1:]
            line_tmp[-1] = line_tmp[-1][:-1]
            tab_id = line_tmp[0]
            col_id = line_tmp[1]
            tab_col = '%s %s' % (tab_id, col_id)
            tab_cols.append(tab_col)
    return tab_cols


# read header name of each column
def read_col_header():
    col_headers = dict()
    prop_dir = os.path.join(t2d_dir, 'property')
    for prop_filename in os.listdir(prop_dir):
        with open(os.path.join(prop_dir, prop_filename)) as f:
            tab_id = prop_filename.split('.csv')[0]
            for line in f.readlines():
                line_tmp = line.strip().split('","')
                line_tmp[0] = line_tmp[0][1:]
                line_tmp[-1] = line_tmp[-1][:-1]
                header = line_tmp[1]
                col_id = line_tmp[3]
                col = '%s %s' % (tab_id, col_id)
                col_headers[col] = header
    return col_headers


# read table column number (order) and table column cells
def read_t2d_cells():
    cols = read_table_cols()
    col_headers = read_col_header()
    col_cells = dict()
    table_dir = os.path.join(t2d_dir, 'tables')
    for col in cols:
        tab_id = col.split(' ')[0]
        col_id = col.split(' ')[1]
        with open(os.path.join(table_dir, ('%s.json' % tab_id))) as f:
            tab_line = f.readline()
            tab_line = tab_line.strip()
            col_contents = tab_line.split("[[")[1].split("]]")[0]
            col_content = col_contents.split('],[')[int(col_id)]
            col_list = col_content.split('","')
            col_list[0] = col_list[0].replace('"', '')
            col_list[-1] = col_list[-1].replace('"', '')
            if col_headers[col] != 'NULL':
                col_list = col_list[1:]
            col_cells[col] = col_list

    return col_cells


# read ground truth label of columns
def read_col_gt():
    col_classes = dict()
    with open(os.path.join(t2d_dir, 'col_class_checked_fg.csv'), 'r') as f:
        for line in f.readlines():
            line_tmp = line.strip().split('","')
            line_tmp[0] = line_tmp[0][1:]
            line_tmp[-1] = line_tmp[-1][:-1]
            tab_id = line_tmp[0]
            col_id = line_tmp[1]
            cls_URI = line_tmp[3]
            ori_cls = cls_URI.split('/')[-1]
            col = '%s %s' % (tab_id, col_id)
            col_classes[col] = [ori_cls]
    return col_classes


# read primary key columns
def primary_key_cols():
    pk_col = set()
    pro_dir = os.path.join(t2d_dir, 'property')
    for file_name in os.listdir(pro_dir):
        with open(os.path.join(pro_dir, file_name), 'r') as f:
            for line in f.readlines():
                line_tmp = line.strip().split('","')
                line_tmp[0] = line_tmp[0][1:]
                line_tmp[-1] = line_tmp[-1][:-1]
                if line_tmp[2] == 'True':
                    col_id = line_tmp[3]
                    tab_id = file_name.split('.csv')[0]
                    col = '%s %s' % (tab_id, col_id)
                    pk_col.add(col)
    return pk_col
