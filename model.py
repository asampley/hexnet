import tensorflow as tf
import os

class Net:
    def __init__(self, params):
        self.session = tf.Session()

        self.WIDTH = params['WIDTH']            # int:    width of state image
        self.HEIGHT = params['HEIGHT']          # int:    height of state image
        self.TIME_STEPS = params['TIME_STEPS']  # int:    number of time steps to evaluate as a state
        self.ACTIONS = params['ACTIONS']        # int:    number of actions available
        LEARNING_RATE = params['LEARNING_RATE'] # float: learning rate for adam optimizer
        self.SAVE_DIR = params['SAVE_DIR']      # string: directory to save summaries and the neural network

        # data, both input and labels
        self.states = tf.placeholder(tf.float32, (None, self.HEIGHT, self.WIDTH, self.TIME_STEPS), 'state') # (batch, height, width, time_step)
        self.poststate = tf.placeholder(tf.float32, (None, self.HEIGHT, self.WIDTH, 1), 'poststate') # (batch, height, width, 1)
        self.action = tf.placeholder(tf.int32, (None,), 'action')
        self.reward = tf.placeholder(tf.float32, (None,), 'reward')
        self.gamma = tf.placeholder(tf.float32, (), 'gamma')

        # create parallel path for getting the value of the subsequent state
        self.states2 = tf.concat([self.states[...,1:], self.poststate], axis=-1)

        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, (), name='keep_prob') # scalar
        
        # global step counter
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # network with path for input states and expected output
        conv1 = self.conv('conv1', self.states, [5,5], [2,2], self.TIME_STEPS, 32)
        conv1_2 = self.conv('conv1', self.states2, [5,5], [2,2], self.TIME_STEPS, 32)
        pool1 = tf.nn.pool(conv1, [2,2], 'MAX', padding='SAME', name='pool1')
        pool1_2 = tf.nn.pool(conv1_2, [2,2], 'MAX', padding='SAME', name='pool1')
        conv2 = self.conv('conv2', pool1, [5,5], [2,2], 32, 64)
        conv2_2 = self.conv('conv2', pool1_2, [5,5], [2,2], 32, 64)
        pool2 = tf.nn.pool(conv2, [2,2], 'MAX', padding='SAME', name='pool2')
        pool2_2 = tf.nn.pool(conv2_2, [2,2], 'MAX', padding='SAME', name='pool2')
        flat1 = tf.reshape(pool2, [-1, pool2.get_shape()[1] * pool2.get_shape()[2] * pool2.get_shape()[3]])
        flat1_2 = tf.reshape(pool2_2, [-1, pool2_2.get_shape()[1] * pool2_2.get_shape()[2] * pool2_2.get_shape()[3]])
        dense1 = tf.layers.dense(flat1, 1024, activation=tf.nn.relu, name='dense1')
        dense1_2 = tf.layers.dense(flat1_2, 1024, activation=tf.nn.relu, name='dense1', reuse=True)
        dense2 = tf.layers.dense(dense1, self.ACTIONS, name='dense2')
        dense2_2 = tf.layers.dense(dense1_2, self.ACTIONS, name='dense2', reuse=True)

        self.output = dense2
        # create target output as the original output, but for the selected actions make it reward + gamma * (value at best action)
        with tf.variable_scope('target_output'):
            self.action_mask = tf.one_hot(self.action, self.ACTIONS, 1.0, 0.0, axis=-1, dtype=tf.float32)
            self.output_increment = -self.output + tf.tile(tf.expand_dims(self.reward + self.gamma * tf.reduce_max(dense2_2, axis=-1), axis=-1), (1, self.ACTIONS))
            self.target_output = tf.stop_gradient(self.output + tf.multiply(self.action_mask, self.output_increment))

        # compute euclidean distance error
        with tf.name_scope("error"):
            self.error = tf.reduce_mean(tf.squared_difference(self.output, self.target_output))

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.error, global_step=self._global_step)
 
        # Make summary op and file
        with tf.name_scope('summary'):
            tf.summary.scalar('error', self.error)
            tf.summary.histogram('target_output', self.target_output)
            tf.summary.histogram('output', self.output)

            self.summaries = tf.summary.merge_all()
            self.summaryFileWriter = tf.summary.FileWriter(self.SAVE_DIR, self.session.graph)

        # Make net saver
        self.saver = tf.train.Saver()

        # initalize variables
        self.session.run([tf.global_variables_initializer()])

    def conv(self, name, input, filter_hw, stride_hw, channels_in, channels_out):
        # create convolution layer
        return tf.layers.conv2d(input, channels_out, filter_hw, stride_hw, padding='SAME', name=name, reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

    def save(self):
        self.saver.save(self.session, os.path.join(self.SAVE_DIR + 'model.ckpt'))

    def restore(self):
        self.saver.restore(self.session, os.path.join(self.SAVE_DIR, 'model.ckpt'))
    
    def train(self, states, next_image, action, reward, gamma, keep_prob = 0.5):
        feed_dict = {
            self.states: states,
            self.poststate: next_image,
            self.action: action,
            self.reward: reward,
            self.gamma: gamma,
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

    def summarize(self, states, next_image, action, reward, gamma):
        feed_dict = {
            self.states: states,
            self.poststate: next_image,
            self.action: action,
            self.reward: reward,
            self.gamma: gamma,
            self.keep_prob: 1.0
        }

        summaries, step = self.session.run(
            [self.summaries, self._global_step],
            feed_dict = feed_dict)
        self.summaryFileWriter.add_summary(summaries, step)
        self.summaryFileWriter.flush()

    def global_step(self):
        return self.session.run([self._global_step])[0]

