import tensorflow as tf
from tensorflow.keras.layers import Wrapper, Conv2D, Conv2DTranspose, Dense, Embedding


class SN(Wrapper):
    # noinspection PyAttributeOutsideInit
    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)

            if type(self.layer) in (Conv2D, Conv2DTranspose, Dense):
                self.w = self.layer.kernel
            elif type(self.layer) == Embedding:
                self.w = self.layer.embeddings
            else:
                raise NotImplementedError

            self.w_shape = self.w.shape.as_list()
            self.u = self.add_weight(shape=[1, self.w_shape[-1]],
                                     initializer=tf.initializers.TruncatedNormal(stddev=0.2),
                                     name='sn_u', trainable=False, dtype=tf.float32,
                                     aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SN, self).build()

    def call(self, inputs, training=None, **kwargs):

        self._compute_weights(training)
        output = self.layer(inputs)
        return output

    def _compute_weights(self, training):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = self.u
        _v = tf.matmul(_u, tf.transpose(w_reshaped))
        _v = _v / (tf.reduce_sum(_v ** 2) ** 0.5 + eps)
        _u = tf.matmul(_v, w_reshaped)
        _u = _u / (tf.reduce_sum(_u ** 2) ** 0.5 + eps)

        if training:
            self.u.assign(_u)
        sigma = tf.matmul(tf.matmul(_v, w_reshaped), tf.transpose(_u))

        if type(self.layer) in (Conv2D, Conv2DTranspose, Dense):
            self.layer.kernel = self.w / sigma
        elif type(self.layer) == Embedding:
            self.layer.embeddings = self.w / sigma
        else:
            raise NotImplementedError
