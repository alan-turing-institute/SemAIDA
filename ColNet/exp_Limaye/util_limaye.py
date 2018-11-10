import os
import sys

limaye_dir = '../Limaye'
if not os.path.exists(limaye_dir):
    print('%s does not exist' % limaye_dir)
    sys.exit(1)


# return cells of a given set of columns
def read_cells_by_cols(cols):
    col_cells = dict()
    for col in cols:
        col_tmp = col.split(' ')
        filename = col_tmp[0]
        col_order = int(col_tmp[1])
        col_f = os.path.join(limaye_dir, 'tables_instance', filename)
        cells = list()
        with open(col_f, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line == '':
                    tmp_line = line.strip().split('","')
                    tmp_line[0] = tmp_line[0][1:]
                    tmp_line[-1] = tmp_line[-1][:-1]
                    cell = tmp_line[col_order]
                    if not cell == '':
                        cells.append(cell)
        col_cells[col] = cells
    return col_cells
