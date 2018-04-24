import tensorflow as tf
import os

class Net:
    def __init__(self, params):
        self.session = tf.Session()

        self.WIDTH = params['WIDTH']            # int:    width of state image
        self.HEIGHT = params['HEIGHT']          # int:    height of state image
        self.TIME_STEPS = params['TIME_STEPS']  # int:    number of time steps sent together in a state
        self.ACTIONS = params['ACTIONS']        # int:    number of actions available
        LEARNING_RATE = params['LEARNING_RATE'] # float: learning rate for adam optimizer
        self.SAVE_DIR = params['SAVE_DIR']      # string: directory to save summaries and the neural network

        # data, both input and labels
        self.states = tf.placeholder(tf.float32, (None, self.WIDTH, self.HEIGHT, self.TIME_STEPS), 'state') # (batch, width, height, time_step)
        self.values = tf.placeholder(tf.float32, (None, self.ACTIONS), 'values') # (batch, action)
        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, (), name='keep_prob') # scalar
        
        # global step counter
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # network
        conv1 = self.conv('conv1', self.states, [5,5], [2,2], self.TIME_STEPS, 32)
        pool1 = tf.nn.pool(conv1, [2,2], 'MAX', padding='SAME', name='pool1')
        conv2 = self.conv('conv2', pool1, [5,5], [2,2], 32, 64)
        pool2 = tf.nn.pool(conv2, [2,2], 'MAX', padding='SAME', name='pool2')
        flat1 = tf.reshape(pool2, [-1, pool2.get_shape()[1] * pool2.get_shape()[2] * pool2.get_shape()[3]])
        dense1 = tf.layers.dense(flat1, 1024, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, self.ACTIONS)

        self.output = dense2

        # compute euclidean distance error
        with tf.name_scope("error"):
            self.error = tf.reduce_mean(tf.squared_difference(self.output, self.values))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.error, global_step=self._global_step)
 
        # Make summary op and file
        with tf.name_scope('summary'):
            tf.summary.scalar('error', self.error)
            tf.summary.histogram('values', self.values)
            tf.summary.histogram('output', self.output)

            self.summaries = tf.summary.merge_all()
            self.summaryFileWriter = tf.summary.FileWriter(self.SAVE_DIR, self.session.graph)

        # Make net saver
        self.saver = tf.train.Saver()

        # initalize variables
        self.session.run([tf.global_variables_initializer()])

    def conv(self, name, input, filter_hw, stride_hw, channels_in, channels_out):
        # create convolution layer
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', filter_hw + [channels_in, channels_out])
            b = tf.get_variable('biases', [channels_out])
            conv = tf.nn.conv2d(input, w, strides=[1] + stride_hw + [1], padding='SAME', name=scope.name)
            conv = tf.nn.relu(conv + b)
        return conv

    def save(self):
        self.saver.save(self.session, os.path.join(self.SAVE_DIR + 'model.ckpt'))

    def restore(self):
        self.saver.restore(self.session, os.path.join(self.SAVE_DIR, 'model.ckpt'))
    
    def train(self, states, values, keep_prob = 0.5):
        feed_dict = {
            self.states: states,
            self.values: values,
            self.keep_prob: keep_prob
        }

        return self.session.run(
            [self.train_fn],
            feed_dict=feed_dict)[0]

    def predict(self, states):
        """
        Returns the outputs and new state of the lstm
        """
        
        feed_dict = {
            self.states: states,
            self.keep_prob: 1.0,
        }

        return self.session.run(
            [self.output],
            feed_dict=feed_dict)[0]

    def summarize(self, states, values):
        feed_dict = {
            self.states: states,
            self.values: values,
            self.keep_prob: 1.0,
        }

        summaries, step = self.session.run(
            [self.summaries, self._global_step],
            feed_dict = feed_dict)
        self.summaryFileWriter.add_summary(summaries, step)
        self.summaryFileWriter.flush()

    def global_step(self):
        return self.session.run([self._global_step])[0]

