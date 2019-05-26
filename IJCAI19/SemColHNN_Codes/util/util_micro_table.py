"""
This file contains functions for micro table construction and encoding
"""
import unicodedata
import numpy as np
from pattern.text.en import tokenize

'''determine the type of a column: string (return False), number (return True)'''


def Is_Number_Col(col_cells):
    threshold = 0.7
    num_cell, non_empty = 0, 0
    for cell in col_cells:
        if cell.strip() != '':
            non_empty += 1

        if Is_Number(cell):
            num_cell += 1

    if non_empty == 0:
        return False
    else:
        return float(num_cell) / float(non_empty) >= threshold


# Return whether a string is a number
def Is_Number(s):
    num_flag = False
    try:
        float(s)
        num_flag = True
    except ValueError:
        pass
    if not num_flag:
        try:
            unicodedata.numeric(s)
            num_flag = True
        except (TypeError, ValueError):
            pass
    return num_flag


''' transform a cell to number (float)
    return 0.0 if the cell does not have number format'''


def To_Number(cell):
    if cell.lower() == 'nan' or cell.lower() == 'inf':
        return 0.0
    try:
        v = float(cell)
        return v
    except ValueError:
        pass
    try:
        v = unicodedata.numeric(cell)
        return v
    except (TypeError, ValueError):
        pass
    return 0.0


''' extract testing samples of a given column
'''


def extract_samples_by_col(columns, col, micro_table_size):
    M, N = micro_table_size
    samples = list()

    tab_name, col_id = col.split(' ')
    col_id = int(col_id)
    N_col_ids = list()
    for i, cells in enumerate(columns):
        if i != col_id:
            N_col_ids.append(i)
        if len(N_col_ids) >= N:
            break

    ''' organize the table as rows (transform), 
        filter out rows whose cell of target column is empty, 
        fill columns with 'NaN' if len(N_col_ids) < N '''
    rows_filter = list()
    for i in range(len(columns[0])):
        if columns[col_id][i].strip() != '':
            row = [columns[col_id][i]]
            for N_col_id in N_col_ids:
                row.append(columns[N_col_id][i])
            if len(N_col_ids) < N:
                row += ['NaN'] * (N - len(N_col_ids))
            rows_filter.append(row)

    ''' slide a window in row dimension, 
                re-organize each table segment as dict,
                append a segment whose length is less than M with rows of 'NaN' (0) '''
    row_num, i = len(rows_filter), 0
    while i < row_num:
        seg = rows_filter[i:(i + M)] if i + M <= row_num else rows_filter[i:row_num]
        seg_len = len(seg)

        col_0 = [seg[j][0] for j in range(seg_len)]
        col_0 += ['NaN'] * (M - seg_len)
        sample = {'col_0': col_0}

        for k in range(N):
            col_k = [seg[j][k + 1] for j in range(seg_len)]
            col_k += ['NaN'] * (M - seg_len)
            sample['col_N_%d' % k] = col_k

        i += 1
        samples.append(sample)

    return samples


''' Preprocess the cell phrase
'''


def cell_phrase_preprocess(cell):
    cell_new = cell.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
        replace('"', ' ').replace("'", ' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
    return cell_new


''' Transform a cell (phrase) to a vector by averaging the word vectors 
'''


def cell_vector_avg(cell, w2v_model):
    vector, n = np.zeros(w2v_model.vector_size), 0
    if not cell == 'NaN':
        ent_n = cell_phrase_preprocess(cell)
        tokenized_line = ' '.join(tokenize(ent_n))
        is_alpha_word_line = [word for word in tokenized_line.lower().split() if word.isalpha()]
        for i, word in enumerate(is_alpha_word_line):
            if word in w2v_model.wv.vocab:
                w_vec = w2v_model.wv[word]
                vector += w_vec.reshape(w2v_model.vector_size)
                n += 1
    return vector if n == 0 else vector / n


''' Transform a cell (phrase) to seq_size vectors 
'''


def cell_vector(cell, w2v_model, seq_size):
    vectors = np.zeros((seq_size, w2v_model.vector_size))
    if not cell == 'NaN':
        ent_n = cell_phrase_preprocess(cell)
        tokenized_line = ' '.join(tokenize(ent_n))
        is_alpha_word_line = [word for word in tokenized_line.lower().split() if word.isalpha()]
        for i, word in enumerate(is_alpha_word_line):
            if i >= seq_size:
                break
            if word in w2v_model.wv.vocab:
                w_vec = w2v_model.wv[word]
                vectors[i] = w_vec
    return vectors


''' Embed a micro table
    Each phrase cell is represented by a vector using averaging of word vectors
    Each number cell is represented by the number vector
'''


def Table_Encode_WV_Avg(micro_table, table_size, w2v_model, use_surrounding_columns=True):
    M, N = table_size
    D = w2v_model.vector_size
    emd = np.zeros((M, (N + 1), D)) if use_surrounding_columns else np.zeros((M, 1, D))

    col_0 = micro_table['col_0']
    for i, cell in enumerate(col_0):
        emd[i][0] = cell_vector_avg(cell, w2v_model)

    if use_surrounding_columns:
        for k in range(N):
            col_k = micro_table['col_N_%d' % k]
            if Is_Number_Col(col_k):
                for i, cell in enumerate(col_k):
                    emd[i][k + 1][0] = To_Number(cell)
            else:
                for i, cell in enumerate(col_k):
                    emd[i][k + 1] = cell_vector_avg(cell, w2v_model)

    return emd


''' Embed a micro table
    Each phrase cell is represented by sequence_size word vectors
    Each number cell is represented by the number vector
'''


def Table_Encode_WV(micro_table, table_size, w2v_model, cell_seq_size, use_surrounding_columns=True):
    M, N = table_size
    D = w2v_model.vector_size
    emd = np.zeros((M, (N + 1), cell_seq_size, D)) if use_surrounding_columns else np.zeros((M, 1, cell_seq_size, D))

    col_0 = micro_table['col_0']
    for i, cell in enumerate(col_0):
        emd[i][0] = cell_vector(cell=cell, w2v_model=w2v_model, seq_size=cell_seq_size)

    if use_surrounding_columns:
        for k in range(N):
            col_k = micro_table['col_N_%d' % k]
            if Is_Number_Col(col_k):
                for i, cell in enumerate(col_k):
                    emd[i][k + 1][0][0] = To_Number(cell)
            else:
                for i, cell in enumerate(col_k):
                    emd[i][k + 1] = cell_vector(cell=cell, w2v_model=w2v_model, seq_size=cell_seq_size)

    return emd


''' Encode a micro_table by 
    first transforming the target column (cells) into a sequence by concatenation
    then encode each word of the sequence to a vector
'''


def Synth_Column_Encode_WV(micro_table, seq_size, w2v_model):
    D = w2v_model.vector_size
    emd = np.zeros((seq_size, 1, D))
    col_0 = micro_table['col_0']
    seq = list()
    for j, cell in enumerate(col_0):
        ent_n = cell_phrase_preprocess(cell)
        tokenized_line = ' '.join(tokenize(ent_n))
        seq += [word for word in tokenized_line.lower().split() if word.isalpha()]
    for j in range(seq_size):
        if j < len(seq):
            if seq[j] in w2v_model.wv.vocab:
                emd[j, 0, :] = w2v_model.wv[seq[j]]
    return emd
