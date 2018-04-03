import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lib

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 20

n_inputs = 28           # MNIST data input 28*28
n_steps = 28            # time step
n_hidden_units = 128     # neurons in hidden layer
n_classes = 10          # MNIST classes 0~9

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_steps])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}

# Define biases
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}

def RNN(X, weights, biases):
    '''

    :param X: (128 batch, 28 steps, 28 inputs)
    :param weights:
    :param biases:
    :return:
    '''
    # hidden layer for input to cell --------------------------------------
    # X ==> (128*28, 28)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 28 inputs)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell ----------------------------------------------------------------
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,
                                             forget_bias=1.0,
                                             state_is_tuple=True)
    # lstm cell is devided into two parts (c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell,
                                        X_in,
                                        initial_state=_init_state,
                                        time_major=False)

    # hidden layer for output as the final results ------------------------
    result = tf.matmul(states[1], weights['out']) + biases['out']
    return result


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % display_step == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1