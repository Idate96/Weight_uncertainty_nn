import tensorflow as tf
from model import Model
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.InteractiveSession()
kl_num = 1

def get_scope_variable(scope_name, var, shape=None, type=None, initializer=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, type, initializer)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

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
        self.sigma_1_prior = 0.001
        self.sigma_2_prior = 10**-7
        self.weights_std = 10**-3
        self.num_batches = self.train_set_size/self.batch_size
        # parameter regulating the mixture of gaussians in the prior
        self.prior_weight = 0.5
        self.num_samples = 3

class BayesNetwork(Model):
    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.build()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.num_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.num_classes), type tf.int32
        """
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
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        with tf.name_scope('dense_layer_1'):
            # weights
            W1_mu = get_scope_variable("W1_mu", "W1_mu", [self.config.num_features, self.config.hidden_dims[0]], tf.float32,
                                    tf.contrib.layers.xavier_initializer())
            # parametrization of the std, such that std > 0 always
            W1_rho = get_scope_variable("W1_rho", "W1_rho", [self.config.num_features, self.config.hidden_dims[0]], tf.float32,
                                     tf.contrib.layers.xavier_initializer())
            W1_sigma = tf.log(1 + tf.exp(W1_rho)) * self.config.weights_std
            W1 = W1_mu + W1_sigma * tf.random_normal([self.config.num_features, self.config.hidden_dims[0]])
            # bias
            b1_mu = get_scope_variable("b1_mu", "b1_mu", [self.config.hidden_dims[0]], tf.float32,
                                    tf.zeros_initializer())
            b1_rho = get_scope_variable("b1_mu", "b1_rho", [self.config.hidden_dims[0]], tf.float32,
                                     tf.zeros_initializer())
            b1_sigma = tf.log(1 + tf.exp(b1_rho)) * self.config.weights_std
            b1 = b1_mu + b1_sigma * tf.random_normal([self.config.hidden_dims[0]])
            h1 = tf.nn.relu(tf.matmul(self.inputs_placeholder, W1) + b1)

        tf.summary.histogram("W1_mu", W1_mu)
        tf.summary.histogram("W1_sigma", W1_sigma)
        tf.summary.histogram("W1", W1)

        with tf.name_scope('dense_layer_2'):
            W2_mu = get_scope_variable("W2_mu", "W2_mu", [self.config.hidden_dims[0], self.config.hidden_dims[1]],
                                    tf.float32, tf.contrib.layers.xavier_initializer())
            self.W2_mu = W2_mu
            W2_rho = get_scope_variable("W2_rho", "W2_rho", [self.config.hidden_dims[0], self.config.hidden_dims[1]], tf.float32,
                                     tf.contrib.layers.xavier_initializer())
            W2_sigma = tf.log(1 + tf.exp(W2_rho)) * self.config.weights_std
            self.W2_sigma = W2_sigma
            W2 = W2_mu + W2_sigma * tf.random_normal([self.config.hidden_dims[0], self.config.hidden_dims[1]])
            # bias
            b2_rho = get_scope_variable("b2_rho", "b2_rho", [self.config.hidden_dims[1]], tf.float32,
                                     tf.contrib.layers.xavier_initializer())
            b2_mu = get_scope_variable("b2_mu", "b2_mu", [self.config.hidden_dims[1]], tf.float32,
                                    tf.zeros_initializer())
            b2_sigma = tf.log(1+tf.exp(b2_rho)) * self.config.weights_std
            b2 = b2_mu + b2_sigma * tf.random_normal([self.config.hidden_dims[1]])
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        tf.summary.histogram("W2_mu", W2_mu)
        tf.summary.histogram("W2_sigma", W2_sigma)
        tf.summary.histogram("W2", W2)

        with tf.name_scope('softmax_layer'):
            W3_mu = get_scope_variable("W3_mu", "W3_mu", [self.config.hidden_dims[1], self.config.num_classes],
                                    tf.float32, tf.contrib.layers.xavier_initializer())
            W3_rho = get_scope_variable("W3_rho", "W3_rho", [self.config.hidden_dims[1], self.config.num_classes],
                                    tf.float32, tf.contrib.layers.xavier_initializer())
            W3_sigma = tf.log(1 + tf.exp(W3_rho)) * self.config.weights_std
            W3 = W3_mu + W3_sigma * tf.random_normal([self.config.hidden_dims[1], self.config.num_classes])
            # bias
            b3_mu = get_scope_variable("b3_mu", "b3_mu", [self.config.num_classes], tf.float32, tf.zeros_initializer())
            b3_rho = get_scope_variable("b3_rho", "b3_rho", [self.config.num_classes], tf.float32, tf.zeros_initializer())
            b3_sigma = tf.log(1 + tf.exp(b3_rho)) * self.config.weights_std
            b3 = b3_mu + b3_sigma * tf.random_normal([self.config.num_classes])
            pred = tf.nn.softmax(tf.matmul(h2, W3) + b3)

        tf.summary.histogram("W3_mu", W3_mu)
        tf.summary.histogram("W3_sigma", W3_sigma)
        tf.summary.histogram("W3", W2)


        cache_weights = ((W1, W1_mu, W1_sigma, b1, b1_mu, b1_sigma),
                         (W2, W2_mu, W2_sigma, b2, b2_mu, b2_sigma),
                         (W3, W3_mu, W3_sigma, b3, b3_mu, b3_sigma))

        return pred, cache_weights

    # def log_bayes_var_post(self, x, mu, sigma):
    #     dist = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)
    #     dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
    #     # print('in log bayes var')
    #     # it gives a weird bug
    #     # prob = tf.log(dist.prob(x))
    #     # prob = x
    #     # print("ext log bayes var")
    #     prob = dist.prob(x)
    #     return prob

    def log_var_post(self, x, mu, sigma):
        # sigma *= self.config.weights_std
        prob = self.gaussianloglikelihood(x, mu, sigma)
        return prob

    def gaussianloglikelihood(self, x, mu, sigma):
        return -0.5 * tf.log(2 * np.pi) - tf.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)

    def log_prior(self, x):
        # mu = tf.zeros_like(x)
        # dist_sigma1 = tf.contrib.distributions.Normal(mu, self.config.sigma_1_prior)
        # dist_sigma2 = tf.contrib.distributions.Normal(mu, self.config.sigma_2_prior)
        # prob = self.config.prior_weight*self.gaussianloglikelihood(x, 0., self.config.sigma_1_prior)\
        #               + (1-self.config.prior_weight)*self.gaussianloglikelihood(x, 0., self.config.sigma_2_prior)
        prob = tf.log((self.config.prior_weight/self.config.sigma_1_prior * tf.exp(-x**2/(2*self.config.sigma_1_prior**2))
                    + (1-self.config.prior_weight)/self.config.sigma_2_prior * tf.exp(-x**2/(2*self.config.sigma_2_prior**2))) +
                      10**-10)
        return prob

    def log_likelihood(self, y, predicted_value):
        # try with simple softmax, if label = 3 select logits[3] as p(D,w)
        prob = self.gaussianloglikelihood(y, predicted_value, 1.)
        # idx_class = tf.cast(tf.argmax(predicted_value, axis=1), tf.int32)
        # print(idx_class.get_shape())
        # print(y.get_shape())
        # prob = tf.log(tf.gather(y, idx_class))
        # print(prob.get_shape())
        return prob

    # def log_prior(self, x):
    #     dist1 = tf.contrib.distributions.Normal(loc=0., scale=self.config.sigma_1_prior)
    #     dist2 = tf.contrib.distributions.Normal(loc=0., scale=self.config.sigma_2_prior)
    #     prob = self.config.prior_weight*dist1.prob(x) + (1-self.config.prior_weight)*dist2.prob(x)
    #     return prob

    # def log_likelihood(self, y, predicted_value):
    #     dist = tf.contrib.distributions.Normal(loc=predicted_value, scale=self.config.sigma_1_prior)
    #     prob = dist.prob(tf.cast(y, tf.float32))
    #     return prob

    def add_loss_op(self):
        log_likelihood, log_qw, log_pw = 0, 0, 0
        log_likelihood_softmax = 0

        for i in range(self.config.num_samples):
            pred, cache_weights = self.forward_pass()
            for W, W_mu, W_sigma, b, b_mu, b_sigma in cache_weights:
                # flatten weights
                # W, W_mu, W_sigma = tf.reshape(W, [-1]), tf.reshape(W_mu, [-1]), tf.reshape(W_sigma, [-1])
                log_qw += tf.reduce_sum(self.log_var_post(W, W_mu, W_sigma)) + \
                        tf.reduce_sum(self.log_var_post(b, b_mu, b_sigma))
                log_pw += tf.reduce_sum(self.log_prior(W)) + tf.reduce_sum(self.log_prior(b))
            log_likelihood += tf.reduce_sum(self.log_likelihood(self.labels_placeholder, pred))
            # a bit weird
            # print(np.shape(pred))
            # print("softmax :", np.shape(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder)))
            # log_likelihood_softmax += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))

        # log_likelihood_softmax /= self.config.num_samples
        # log_likelihood /= self.config.num_samples
        log_likelihood /= self.config.num_samples
        self.log_likelihood_value = log_likelihood
        log_pw /= self.config.num_samples
        self.log_pw = log_pw
        log_qw /= self.config.num_samples
        self.log_qw = log_qw
        tf.summary.scalar("log_prior", log_qw)
        tf.summary.scalar("log_post", log_qw)
        tf.summary.scalar("log_likelyhood", log_likelihood)
        # tf.summary.scalar("log_likelyhood_softmax", log_likelihood_softmax)
        # log_lik_print = tf.Print(log_likelihood, [log_likelihood], message="This is a: ")
        # b = tf.add(log_lik_print, log_lik_print).eval()

        # log_likelihood = 0
        # print("value log likelyhood", log_likelihood.eval())
        # print("value log prior ", log_pw.eval())
        # print("value posterior ", log_qw.eval())
        # check if you have to minimize or maximize it
        # the loss is fucking wrong
        loss = 1/self.config.train_set_size*(1/self.config.num_batches * kl_num * (log_qw - log_pw)
                                            - log_likelihood)
        tf.summary.scalar("loss", loss)
        return loss

    def add_training_op(self, loss):
        train_op = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(loss)
        return train_op

    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        # print("Prediction shape : ", np.shape(predictions))
        return predictions

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
        return accuracy

    def fit(self, sess, data):
        global kl_num
        num_batches = int(self.config.train_set_size / self.config.batch_size)
        global_step = 0
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("summary/400_lr3_sigma37-0", sess.graph)
        for epoch in range(self.config.num_epochs):
            for i in range(num_batches):
                kl_num = 2**(num_batches - i)/(2**num_batches-1)
                batch = data.train.next_batch(100)
                loss = self.train_on_batch(sess, batch[0], batch[1])
                if i % 200 == 0:
                    loss, summary = sess.run([self.add_loss_op(), summary_op], feed_dict=self.create_feed_dict(batch[0], batch[1]))
                    summary_writer.add_summary(summary, global_step=global_step)

            global_step += 1
            print("epoch %d, loss %g " % (epoch, self.train_on_batch(sess, data.train.images, data.train.labels)))
            train_accuracy = self.accuracy().eval(feed_dict={
                self.inputs_placeholder: batch[0], self.labels_placeholder: batch[1]})
            print('epoch %d, training accuracy %f' % (epoch, train_accuracy))

            print('test accuracy :', self.accuracy().eval(feed_dict={self.inputs_placeholder: data.test.images,
                                           self.labels_placeholder: data.test.labels}))


def train_model():
    config = Config(28**2, [400, 400], 10, 100, 10**-4, 100)
    model = BayesNetwork(config)
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as session:
        session.run(init)
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        model.fit(session, mnist)

if __name__ == '__main__':
    train_model()