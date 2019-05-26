import os
import datetime
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from util_rnn import attention, batch_iter


def hnn_train(x_train, y_train, FLAGS, w2v_model, hnn_dir):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():

            hnn = HNN(micro_table_size=FLAGS.micro_table_size, sequence_length=FLAGS.cell_seq_size,
                      num_classes=y_train.shape[1], channel_num=w2v_model.vector_size,
                      rnn_hidden_size=FLAGS.rnn_hidden_size, attention_size=FLAGS.attention_size,
                      col_filters=FLAGS.col_filters, row_filters=FLAGS.row_filters,
                      num_filters=FLAGS.num_filters, num_cell_features=FLAGS.num_cell_features)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(hnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", hnn.loss)
            acc_summary = tf.summary.scalar("accuracy", hnn.accuracy)
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(hnn_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(hnn_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(train_x_batch, train_y_batch):
                feed_dict = {
                    hnn.input_x: train_x_batch,
                    hnn.input_y: train_y_batch,
                    hnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, hnn.loss, hnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % FLAGS.evaluate_every == 0:
                    print("\t {}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def train_eva(train_x_all, train_y_all):
                feed_dict = {
                    hnn.input_x: train_x_all,
                    hnn.input_y: train_y_all,
                    hnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                loss, accuracy = sess.run([hnn.loss, hnn.accuracy], feed_dict)
                print("\t finally, train loss {:g}, train acc {:g}".format(loss, accuracy))

            batches = batch_iter(list(zip(x_train, y_train)), FLAGS.num_epochs, FLAGS.batch_size)
            current_step = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

            train_eva(x_train, y_train)

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("\t Saved model checkpoint to {}\n".format(os.path.basename(path)))


''' Predict with HNN 
    input: test_x (each row represents one sample vector)
    output: test_p (an array, each item represents the score of one sample)
            test_fc (output of the FC layer)
'''
def hnn_predict(test_x, hnn_dir, need_fc_out):
    checkpoint_dir = os.path.join(hnn_dir, 'checkpoints')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            if need_fc_out:
                FC_out = graph.get_operation_by_name("output/FC_out").outputs[0]
                test_y, test_fc_out = sess.run([scores, FC_out], {input_x: test_x, dropout_keep_prob: 1.0})
            else:
                test_y = sess.run(scores, {input_x: test_x, dropout_keep_prob: 1.0})
                test_fc_out = None

    return test_y, test_fc_out


''' 
    Hybrid Neural Network:
        + phrase (cell) encoding with RNN and an attention layer
        + column feature with a Conv layer
        + a fully connected layer
    col_filters and row_filters can be set to [], so as to disabling col or row features
    num_cell_features can be set to 0, so as to disabling cell features
    regarding no col/row conv features, no cell features, please use AttRNN in util_rnn.py
'''


class HNN(object):

    def __init__(self, micro_table_size, sequence_length, num_classes, channel_num, rnn_hidden_size, attention_size,
                 col_filters, row_filters, num_filters, num_cell_features):
        M, N = micro_table_size
        rnn_dim = 2 * rnn_hidden_size

        self.input_x = tf.placeholder(tf.float32, [None, M, (N + 1), sequence_length, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # encode each cell of the target column
        # by bidirectional RNN, attention layer, dropout layer
        col_att_drops = list()
        for i in range(M):
            with tf.variable_scope('col_rnn_%d' % i):
                rnn_outputs, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                        inputs=self.input_x[:, i, 0, :, :], dtype=tf.float32)
            with tf.name_scope('col_attention_layer_%d' % i):
                att_output, alphas = attention(rnn_outputs, attention_size, return_alphas=True)
            with tf.name_scope("col_dropout_%d" % i):
                att_drop = tf.nn.dropout(att_output, self.dropout_keep_prob)
                col_att_drops.append(att_drop)
        self.col_atts = tf.reshape(tf.stack(col_att_drops, axis=1), [-1, M, 1, rnn_dim])

        row_att_drops = list()
        if len(row_filters) > 0:
            for i in range(N + 1):
                with tf.variable_scope('row_rnn_%d' % i):
                    rnn_outputs, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                            inputs=self.input_x[:, 0, i, :, :], dtype=tf.float32)
                with tf.name_scope('row_attention_layer_%d' % i):
                    att_output, alphas = attention(rnn_outputs, attention_size, return_alphas=True)
                with tf.name_scope("row_dropout_%d" % i):
                    att_drop = tf.nn.dropout(att_output, self.dropout_keep_prob)
                    row_att_drops.append(att_drop)
            self.row_atts = tf.reshape(tf.stack(row_att_drops, axis=2), [-1, 1, N + 1, rnn_dim])

        # Conv layer
        # Get col features over the target column, i.e., col_filters
        conv_col_pooled = []
        for i, f in enumerate(col_filters):
            x_col_0 = self.col_atts
            with tf.name_scope("conv-col-%d" % f):
                f_shape = [f, 1, rnn_dim, num_filters]
                Conv_W = tf.Variable(tf.truncated_normal(f_shape, stddev=0.1), name="conv_%d_W" % f)
                Conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_%d_b" % f)
                conv = tf.nn.conv2d(x_col_0, Conv_W, strides=[1, 1, 1, 1], padding="VALID", name="conv_%d" % f)
                h = tf.nn.relu(tf.nn.bias_add(conv, Conv_b), name="relu_%d" % f)
                pooled = tf.nn.max_pool(h, ksize=[1, M - f + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool_%s" % f)
                conv_col_pooled.append(pooled)

        # Get col (context) features, i.e., row_filters
        conv_row_pooled = []
        for i, f in enumerate(row_filters):
            x_row_0 = self.row_atts
            with tf.name_scope('conv-row-%d' % f):
                f_shape = [1, f, rnn_dim, num_filters]
                Conv_W = tf.Variable(tf.truncated_normal(f_shape, stddev=0.1), name="conv_%d_W" % f)
                Conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_%d_b" % f)
                conv = tf.nn.conv2d(x_row_0, Conv_W, strides=[1, 1, 1, 1], padding="VALID", name="conv_%d" % f)
                h = tf.nn.relu(tf.nn.bias_add(conv, Conv_b), name="relu_%d" % f)
                pooled = tf.nn.max_pool(h, ksize=[1, 1, N + 1 - f + 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool_%d" % f)
                conv_row_pooled.append(pooled)

        # Feature of the cell
        if num_cell_features > 0:
            with tf.name_scope('cell'):
                Cell_W = tf.get_variable("Cell_W", shape=[rnn_dim, num_cell_features],
                                         initializer=tf.contrib.layers.xavier_initializer())
                Cell_b = tf.Variable(tf.constant(0.1, shape=[num_cell_features]), name="Cell_b")
                self.cell_f = tf.nn.xw_plus_b(self.col_atts[:, 0, 0, :], Cell_W, Cell_b, name='cell_feature')

        # col/row Conv features + cell features
        num_filters_total = num_filters * (len(conv_col_pooled) + len(conv_row_pooled))
        if num_filters_total > 0 and num_cell_features > 0:
            self.h_pool = tf.concat(conv_col_pooled + conv_row_pooled, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            self.feature = tf.concat([self.cell_f, self.h_pool_flat], 1)
            num_features = num_filters_total + num_cell_features

        # cell features
        elif num_filters_total == 0 and num_cell_features > 0:
            self.feature = self.cell_f
            num_features = num_cell_features

        # col/row Conv features
        else:
            self.h_pool = tf.concat(conv_col_pooled, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            self.feature = self.h_pool_flat
            num_features = num_filters_total

        with tf.name_scope("dropout"):
            self.feature_drop = tf.nn.dropout(self.feature, self.dropout_keep_prob)

        # FC layer
        with tf.name_scope("output"):
            FC_W = tf.get_variable("FC_W", shape=[num_features, num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="FC_b")
            self.fc_out = tf.nn.xw_plus_b(self.feature_drop, FC_W, FC_b, name="FC_out")
            self.scores = tf.nn.softmax(self.fc_out, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
