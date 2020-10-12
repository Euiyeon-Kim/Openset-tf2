from keras.models import Model
from keras.layers import Input, Flatten, Conv2D, LeakyReLU
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import multi_gpu_model

from models.layers import sn_conv_blocks, dense_branches


class SpecNormClassifier:
    def __init__(self, config):
        self.config = config

    def define_model(self):
        input_layer = Input(shape=self.config.input_shape)

        shared_conv = sn_conv_blocks(channels=self.config.shared_conv_channels,
                                     kernel_size=self.config.kernel_size,
                                     strides=self.config.strides,
                                     activation=self.config.activation,
                                     name='shared')(input_layer)

        conv_flatten = Flatten()(shared_conv)

        out = dense_branches(units=self.config.dense_branch_units,
                             output_num=self.config.num_classes,
                             activation=self.config.activation,
                             name='class'
                             )(conv_flatten)

        return Model(input_layer, out, name='spec_norm_classifier')

    def build_model(self):
        model = self.define_model()
        parallel_model = multi_gpu_model(model, gpus=self.config.n_gpus) if self.config.n_gpus > 1 else model
        optimizer = Adam(self.config.lr, self.config.beta1, self.config.beta2, decay=0.01/30000)
        parallel_model.compile(optimizer=optimizer,
                               loss={"class_output": categorical_crossentropy},
                               metrics={"class_output": "accuracy"})
        return parallel_model
