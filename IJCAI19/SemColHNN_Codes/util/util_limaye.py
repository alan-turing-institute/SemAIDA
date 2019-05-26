"""
This file contains functions to access data of Limaye
"""
import os
import sys
import json

limaye_dir = 'Limaye'
if not os.path.exists(limaye_dir):
    print('%s does not exist' % limaye_dir)
    sys.exit(1)

# read the content (relation) of a table as a 2-dim array
def read_limaye_columns(tab_id):
    table_dir = os.path.join(limaye_dir, 'table_instance_json')
    with open(os.path.join(table_dir, ('%s.json' % tab_id))) as f:
        line = f.readline()
        contents = json.loads(line.decode("utf-8", "ignore"))
        contents = contents['contents']
        table = list()
        row_num = len(contents)
        if row_num > 0:
            col_num = len(contents[0])
            for col_id in range(col_num):
                col = list()
                for row_id in range(row_num):
                    col.append(contents[row_id][col_id]['data'])
                table.append(col)
    return table
