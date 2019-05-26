import os
import datetime
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn


def rnn_train(x_train, y_train, FLAGS, w2v_model, rnn_dir):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = AttRNN(sequence_length=FLAGS.cell_seq_size, num_classes=y_train.shape[1],
                         channel_num=w2v_model.vector_size, rnn_hidden_size=FLAGS.rnn_hidden_size,
                         attention_size=FLAGS.attention_size)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(rnn.loss)
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
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(rnn_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(rnn_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(train_x_batch, train_y_batch):
                feed_dict = {
                    rnn.input_x: train_x_batch,
                    rnn.input_y: train_y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % FLAGS.evaluate_every == 0:
                    print("\t {}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def train_eva(train_x_all, train_y_all):
                feed_dict = {
                    rnn.input_x: train_x_all,
                    rnn.input_y: train_y_all,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                loss, accuracy = sess.run([rnn.loss, rnn.accuracy], feed_dict)
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


''' Predict with RNN 
    input: test_x (each row represents one sample vector)
    output: test_p (an array, each item represents the score of one sample)
'''
def rnn_predict(test_x, rnn_dir):
    checkpoint_dir = os.path.join(rnn_dir, 'checkpoints')
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


'''
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
         for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
'''
def attention(inputs, attention_size, time_major=False, return_alphas=False):
    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output if not return_alphas else output, alphas


''' 
    + phrase (cell) encoding with RNN and an attention layer
    + a fully connected layer
'''
class AttRNN(object):

    def __init__(self, sequence_length, num_classes, channel_num, rnn_hidden_size, attention_size):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Bidirectional RNN
        self.rnn_outputs, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                     inputs=self.input_x, dtype=tf.float32)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            self.att_output, alphas = attention(self.rnn_outputs, attention_size, return_alphas=True)
            tf.summary.histogram('alphas', alphas)

        # Dropout layer
        with tf.name_scope("dropout"):
            self.att_drop = tf.nn.dropout(self.att_output, self.dropout_keep_prob)

        # FC layer
        with tf.name_scope("output"):
            FC_W = tf.get_variable("FC_W", shape=[rnn_hidden_size * 2, num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="FC_b")
            self.fc_out = tf.nn.xw_plus_b(self.att_drop, FC_W, FC_b, name="FC_out")
            self.scores = tf.nn.softmax(self.fc_out, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
