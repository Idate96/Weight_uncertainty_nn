import tensorflow as tf
from model import Model
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.InteractiveSession()

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self, num_features, hidden_dims, num_classes, num_epochs, learning_rate, batch_size):
        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_set_size = 55000
        self.batch_size = batch_size
        self.sigma_1_prior = 10**-2
        self.sigma_2_prior = 10**-3
        # parameter regulating the mixture of gaussians in the prior
        self.prior_weight = 1
        self.num_samples = 10

class BasicNN(Model):

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.build()

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(shape=[None, self.config.num_features], dtype=tf.float32)
        self.labels_placeholder = tf.placeholder(shape=[None, self.config.num_classes], dtype=tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def forward_pass(self):
        with tf.name_scope("layer1"):
            dense_1 = tf.layers.dense(inputs=self.inputs_placeholder, units = self.config.hidden_dims[0],
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=tf.nn.relu)
        with tf.name_scope("layer2"):
            dense_2 = tf.layers.dense(inputs=dense_1, units = self.config.hidden_dims[1],
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=tf.nn.relu)
        with tf.name_scope("layer3"):
            preds = tf.layers.dense(inputs=dense_2, units=self.config.num_classes,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        return preds

    def add_loss_op(self, pred):
        # labels = tf.one_hot(self.labels_placeholder, self.config.num_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.AdagradOptimizer(self.config.learning_rate).minimize(loss)
        return train_op

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.config.num_epochs):
                for _ in range(int(self.config.train_set_size / self.config.batch_size)):
                    batch = data.train.next_batch(100)
                    feed_dict = self.create_feed_dict(batch[0], batch[1])
                    _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                train_accuracy = self.accuracy().eval(feed_dict={
                    self.inputs_placeholder: batch[0], self.labels_placeholder: batch[1]})
                print('epoch %d, training accuracy %g' % (epoch, train_accuracy))

            print('test accuracy ', self.accuracy().eval(feed_dict={
                self.inputs_placeholder: data.test.images, self.labels_placeholder: data.test.labels}))



if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    config = Config(28 ** 2, [500, 800], 10, 20, 10 ** -2, 100)
    model = BasicNN(config)
    model.train(mnist)
