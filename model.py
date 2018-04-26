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
        self.states = tf.placeholder(tf.float32, (None, self.HEIGHT, self.WIDTH, self.TIME_STEPS + 1), 'state') # (batch, height, width, time_step + 1)
        self.action = tf.placeholder(tf.int32, (None,), 'action')
        self.reward = tf.placeholder(tf.float32, (None,), 'reward')
        self.gamma = tf.placeholder(tf.float32, (), 'gamma')
        self.postterminal = tf.placeholder(tf.bool, (None,), 'postterminal')

        # create parallel path for getting the value of the subsequent state
        self.states1 = self.states[...,:-1]
        self.states2 = self.states[...,1:]

        # Set variable for dropout of each layer
        self.keep_prob = tf.placeholder(tf.float32, (), name='keep_prob') # scalar
        
        # global step counter
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        # network with path for input states and expected output
        self.path_output = []
        for path_states in (self.states1, self.states2):
            conv1 = self.conv('conv1', path_states, [8,8], [4,4], self.TIME_STEPS, 16)
            conv2 = self.conv('conv2', conv1, [4,4], [2,2], 16, 32)
            flat1 = tf.layers.flatten(conv2)
            dense1 = tf.layers.dense(flat1, 256, activation=tf.nn.relu, name='dense1', reuse=tf.AUTO_REUSE)
            dense2 = tf.layers.dense(dense1, self.ACTIONS, name='dense2', reuse=tf.AUTO_REUSE)
            self.path_output += [dense2]

        self.output = self.path_output[0]
        self.next_output = self.path_output[1]

        # create target output as the original output, but for the selected actions make it reward + gamma * (value at best action)
        with tf.variable_scope('target_output'):
            self.action_mask = tf.one_hot(self.action, self.ACTIONS, 1.0, 0.0, axis=-1, dtype=tf.float32)
            self.future_reward = tf.multiply(tf.cast(self.postterminal, tf.float32), self.gamma * tf.reduce_max(self.next_output, axis=-1))
            self.output_increment = -self.output + tf.tile(tf.expand_dims(self.reward + self.future_reward, axis=-1), (1, self.ACTIONS))
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
    
    def train(self, states, action, reward, next_terminal, gamma, keep_prob = 0.5):
        feed_dict = {
            self.states: states,
            self.action: action,
            self.reward: reward,
            self.postterminal: next_terminal,
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

    def summarize(self, states, action, reward, next_terminal, gamma):
        feed_dict = {
            self.states: states,
            self.action: action,
            self.reward: reward,
            self.postterminal: next_terminal,
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

