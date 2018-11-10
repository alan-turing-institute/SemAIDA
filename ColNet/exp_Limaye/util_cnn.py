import tensorflow as tf
import random
import numpy as np
from pattern.text.en import tokenize


def permutation_cells2synthetic_columns(col_cells):
    cell_units = list()
    if len(col_cells) >= 2:
        for ci in col_cells:
            for cj in col_cells:
                if not ci == cj:
                    cell_units.append([ci, cj])
    elif len(col_cells) == 1:
        cell_units = [col_cells[0], 'NaN']
    return cell_units


def ordered_cells2synthetic_columns(col_cells, synthetic_column_size):
    cell_units = list()
    if len(col_cells) >= synthetic_column_size:
        for cell_i in range(len(col_cells) - synthetic_column_size + 1):
            cell_unit = col_cells[cell_i:cell_i+synthetic_column_size]
            cell_units.append(cell_unit)
    else:
        cell_unit = col_cells + ['NaN'] * (len(col_cells) - synthetic_column_size)
        cell_units.append(cell_unit)
    return cell_units


def random_cells2synthetic_columns(col_cells, synthetic_column_size, synthetic_column_num):
    cell_units = list()
    if len(col_cells) >= synthetic_column_size:
        for i in range(synthetic_column_num):
            cell_units.append(random.sample(col_cells, synthetic_column_size))
    else:
        for i in range(synthetic_column_num):
            cell_unit = list()
            for j in range(synthetic_column_size):
                cell_unit += random.sample(col_cells, 1)
            cell_units.append(cell_unit)

    return cell_units


def generate_synthetic_columns(entities, synthetic_column_size):
    ent_units = list()
    if len(entities) >= synthetic_column_size:
        for i, ent in enumerate(entities):
            unit = random.sample(entities[0:i] + entities[(i + 1):], synthetic_column_size - 1)
            unit.append(ent)
            ent_units.append(unit)
    else:
        unit = entities + ['NaN'] * (len(entities) - synthetic_column_size)
        ent_units.append(unit)
    return ent_units


def synthetic_columns2sequence(ent_units, sequence_size):
    word_seq = list()
    for ent in ent_units:
        ent_n = ent.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
            replace('"', ' ').replace("'", ' ')
        tokenized_line = ' '.join(tokenize(ent_n))
        is_alpha_word_line = [word for word in tokenized_line.lower().split() if word.isalpha()]
        word_seq += is_alpha_word_line
    if len(word_seq) >= sequence_size:
        return word_seq[0:sequence_size]
    else:
        return word_seq + ['NaN'] * (sequence_size - len(word_seq))


def sequence2matrix(word_seq, sequence_size, w2v_model):
    ent_v = np.zeros((sequence_size, w2v_model.vector_size, 1))
    for i, word in enumerate(word_seq):
        if not word == 'NaN' and word in w2v_model.wv.vocab:
            w_vec = w2v_model.wv[word]
            ent_v[i] = w_vec.reshape((w2v_model.vector_size, 1))
    return ent_v


class SyntheticColumnCNN(object):
    """
    A CNN for classification of entity units.
    Uses a Conv layer with multiple filters, followed by a max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, embedding_size, channel_num, filter_sizes, num_filters):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        # l2_loss = tf.constant(0.0)
        # l2_reg_lambda = 0.0

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-max_pool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                Conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Conv_W")
                Conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="Conv_b")
                conv = tf.nn.conv2d(
                    self.input_x,
                    Conv_W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, Conv_b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            FC_W = tf.get_variable("FC_W",
                                   shape=[num_filters_total, num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="FC_b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, FC_W, FC_b, name="scores")
            self.probabilities = tf.nn.softmax(self.scores, name='probabilities')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            # self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
