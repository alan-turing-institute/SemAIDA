import os
import datetime
import tensorflow as tf
import numpy as np


def cnn_train(x_train, y_train, FLAGS, w2v_model, cnn_dir, v_train):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = MicroTableCNN(micro_table_size=FLAGS.micro_table_size, num_classes=y_train.shape[1],
                                channel_num=w2v_model.vector_size, col_filters=FLAGS.col_filters,
                                row_filters=FLAGS.row_filters, cell_prop_size=FLAGS.cell_prop_size,
                                FC_size=FLAGS.FC_size,
                                num_filters=FLAGS.num_filters, prop2vec_dim=FLAGS.prop2vec_dim,
                                use_property_vector=FLAGS.use_property_vector)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
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
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(cnn_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(cnn_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(feeds):
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feeds)
                time_str = datetime.datetime.now().isoformat()
                if step % FLAGS.evaluate_every == 0:
                    print("\t {}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def train_eva(feeds):
                loss, accuracy = sess.run([cnn.loss, cnn.accuracy], feeds)
                print("\t finally, train loss {:g}, train acc {:g}".format(loss, accuracy))

            if FLAGS.use_property_vector:
                batches = batch_iter(list(zip(x_train, y_train, v_train)), FLAGS.num_epochs, FLAGS.batch_size)
                current_step = 0
                for batch in batches:
                    x_batch, y_batch, v_batch = zip(*batch)
                    feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.input_v: v_batch,
                                 cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                    train_step(feed_dict)
                    current_step = tf.train.global_step(sess, global_step)
            else:
                batches = batch_iter(list(zip(x_train, y_train)), FLAGS.num_epochs, FLAGS.batch_size)
                current_step = 0
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch,
                                 cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                    train_step(feed_dict)
                    current_step = tf.train.global_step(sess, global_step)

            if FLAGS.use_property_vector:
                feed_dict = {cnn.input_x: x_train, cnn.input_y: y_train, cnn.input_v: v_train,
                             cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            else:
                feed_dict = {cnn.input_x: x_train, cnn.input_y: y_train,
                             cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            train_eva(feed_dict)

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("\t Saved model checkpoint to {}\n".format(os.path.basename(path)))


''' Predict with CNN
    input: test_x (each row represents one sample vector)
    output: test_p (an array, each item represents the score of one sample)
'''


def cnn_predict(test_x, cnn_dir):
    checkpoint_dir = os.path.join(cnn_dir, 'checkpoints')
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

            test_y = sess.run(scores, {input_x: test_x, dropout_keep_prob: 1.0})

    return test_y


''' Generate batches of the samples
    In each epoch, samples are traversed one time batch by batch
'''


def batch_iter(data, num_epochs, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            batch_shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[batch_shuffle_indices]
        else:
            shuffled_data = data

        if num_batches > 0:
            for batch_num in range(num_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
        else:
            yield shuffled_data


''' A CNN for classification of micro tables:
    + a Conv layer with multiple filters
        1) filters over the target column with size of e.g., 2*1*D and 3*1*D
        2) filter over the combination of target column and a surrounding column, with size of 1*2*D
    + a max-pooling layer over features (aggregation on row dimension)
    + a fully connected layer
    + a softmax layer for classification
'''


class MicroTableCNN(object):

    def __init__(self, micro_table_size, num_classes, channel_num, col_filters, row_filters, cell_prop_size, FC_size,
                 num_filters, prop2vec_dim, use_property_vector):
        M, N = micro_table_size

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, M, (N + 1), channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        if use_property_vector:
            self.input_v = tf.placeholder(tf.float32, [None, prop2vec_dim], name="input_v")

        # Get (coherence) features over the target column, i.e., col_filters
        pooled_outputs_col = []
        for i, f in enumerate(col_filters):
            x_col_0 = self.input_x[:, :, 0:1, :]
            with tf.name_scope("col-%d" % f):
                f_shape = [f, 1, channel_num, num_filters]
                Conv_W = tf.Variable(tf.truncated_normal(f_shape, stddev=0.1), name="conv_%d_W" % f)
                Conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_%d_b" % f)
                conv = tf.nn.conv2d(x_col_0, Conv_W, strides=[1, 1, 1, 1], padding="VALID", name="conv_%d" % f)
                h = tf.nn.relu(tf.nn.bias_add(conv, Conv_b), name="relu_%d" % f)
                pooled = tf.nn.max_pool(h, ksize=[1, M - f + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool_%s" % f)
                pooled_outputs_col.append(pooled)

        # Get col (context) features, i.e., row_filters
        pooled_outputs_row = []
        for i, f in enumerate(row_filters):
            x_row_0 = self.input_x[:, 0:1, :, :]
            with tf.name_scope('row-%d' % f):
                f_shape = [1, f, channel_num, num_filters]
                Conv_W = tf.Variable(tf.truncated_normal(f_shape, stddev=0.1), name="conv_%d_W" % f)
                Conv_b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_%d_b" % f)
                conv = tf.nn.conv2d(x_row_0, Conv_W, strides=[1, 1, 1, 1], padding="VALID", name="conv_%d" % f)
                h = tf.nn.relu(tf.nn.bias_add(conv, Conv_b), name="relu_%d" % f)
                pooled = tf.nn.max_pool(h, ksize=[1, 1, N + 1 - f + 1, 1], strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool_%d" % f)
                pooled_outputs_row.append(pooled)

        # FC layer for features from cell + prop2vec
        with tf.name_scope('cell_prop'):
            if use_property_vector:
                self.cell_prop_input = tf.concat([self.input_x[:, 0, 0, :], self.input_v], 1)
                num_cell_prop = channel_num + prop2vec_dim
            else:
                self.cell_prop_input = self.input_x[:, 0, 0, :]
                num_cell_prop = channel_num
            Cell_Prop_W = tf.get_variable("cell_prop_W", shape=[num_cell_prop, cell_prop_size],
                                          initializer=tf.contrib.layers.xavier_initializer())
            Cell_Prop_b = tf.Variable(tf.constant(0.1, shape=[cell_prop_size]), name="cell_prop_b")
            self.cell_prop_out = tf.nn.xw_plus_b(self.cell_prop_input, Cell_Prop_W, Cell_Prop_b,
                                                 name='cell_prop_out')
            # self.cell_prop_out = tf.nn.relu_layer(self.cell_prop_input, Cell_Prop_W, Cell_Prop_b,
            #                                       name='cell_prop_out')
            # self.reg1 = tf.nn.l2_loss(Cell_Prop_W)

        num_filters_total = num_filters * (len(pooled_outputs_col) + len(pooled_outputs_row))
        if num_filters_total > 0:
            self.h_pool = tf.concat(pooled_outputs_col + pooled_outputs_row, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            self.features = tf.concat([self.cell_prop_out, self.h_pool_flat], 1)
            num_features = num_filters_total + cell_prop_size
        else:
            self.features = self.cell_prop_out
            num_features = cell_prop_size

        with tf.name_scope("dropout"):
            self.f_drop = tf.nn.dropout(self.features, self.dropout_keep_prob)

        with tf.name_scope("FC"):
            FC_W = tf.get_variable("FC_W", shape=[num_features, FC_size],
                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_b = tf.Variable(tf.constant(0.1, shape=[FC_size]), name="FC_b")
            fc_out = tf.nn.relu_layer(self.features, FC_W, FC_b)
            self.fc_out = tf.nn.dropout(fc_out, self.dropout_keep_prob)

        with tf.name_scope("output"):
            out_W = tf.get_variable("out_W", shape=[FC_size, num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
            out_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="out_b")
            self.out = tf.nn.xw_plus_b(self.fc_out, out_W, out_b, name="out")
            self.scores = tf.nn.softmax(self.out, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.input_y))

        # Calculate accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
