"""
This file contains functions to access data of T2D
"""

import os
import sys
import json

t2d_dir = 'T2D'
if not os.path.exists(t2d_dir):
    print('%s does not exist' % t2d_dir)
    sys.exit(1)


# read the content (relation) of a table as a 2-dim array
def read_t2d_columns(tab_id):
    table_dir = os.path.join(t2d_dir, 'tables')
    with open(os.path.join(table_dir, ('%s.json' % tab_id))) as f:
        line = f.readline()
        content = json.loads(line.decode("utf-8", "ignore"))
        relation = content['relation']
        table = list()
        for rel in relation:
            table.append(rel[1:])
    return table
