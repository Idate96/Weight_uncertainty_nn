class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def forward_pass(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, *args):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, *args):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        # w2_mu, w2_sigma = sess.run([self.W2_mu, self.W2_sigma], feed_dict=feed)
        # print("w2 mu " + 40 * "=", w2_mu)
        # print("w2 sigma " + 40 * "-", w2_sigma)
        # log_likelyhood_value, logqw, logpw = sess.run([self.log_likelihood_value, self.log_qw, self.log_pw], feed_dict=feed)
        # print("log likelyhood : {0} \n logqw : {1} \n logpw {2}".format(log_likelyhood_value, logqw, logpw))
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred, self.cached_weights = self.forward_pass()
        self.loss = self.add_loss_op()
        self.train_op = self.add_training_op(self.loss)
