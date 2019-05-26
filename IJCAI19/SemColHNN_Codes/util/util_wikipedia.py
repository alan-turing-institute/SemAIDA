"""
This file contains functions to access data of Limaye
"""
import os
import sys
import re
import csv

wikipedia_dir = 'Wikipedia'
if not os.path.exists(wikipedia_dir):
    print('%s does not exist' % wikipedia_dir)
    sys.exit(1)

# read the content (relation) of a table as a 2-dim array
def read_wikipedia_columns(tab_id):
    rows = list()
    table_dir = os.path.join(wikipedia_dir, 'select_tables')
    f = open(os.path.join(table_dir, ('%s.csv' % tab_id)))
    csv_r = csv.reader(f, delimiter=',', quotechar='"')
    for i, line in enumerate(csv_r):

        # the size of the first line (often the header)
        if i == 0:
            row_len = len(line)

        # from the second line, pad with zero (cut) if the line is shorter (longer) than the first line
        # ignore the empty line
        else:
            line2 = list()
            for item in line:
                item = item.replace('\/', ' ')
                for bracket in re.findall('\[([0-9]*?)\]', item):
                    item = item.replace('['+bracket+']', ' ')
                try:
                    item = item.decode('unicode-escape').encode('ascii', 'ignore')
                except UnicodeDecodeError:
                    item = ''
                line2.append(item)

            m = len(line2)
            if m > 0:
                if m >= row_len:
                    row = line2[0:row_len]
                    rows.append(row)
                else:
                    row = line2 + [''] * (row_len - m)
                    rows.append(row)
    f.close()
    cols = list()
    for r in range(row_len):
        col = list()
        for row in rows:
            col.append(row[r])
        cols.append(col)

    return cols
