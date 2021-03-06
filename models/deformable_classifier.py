from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import multi_gpu_model

from models.layers import conv2d, conv_blocks, deform_conv_block, dense_branches


class DeformableClassifier:
    def __init__(self, config):
        self.config = config

    def define_model(self):
        input_layer = Input(shape=self.config.input_shape)

        shared_conv = conv_blocks(channels=self.config.deform_conv_channels[:3],
                                  kernel_size=self.config.kernel_size,
                                  strides=self.config.strides,
                                  activation=self.config.activation,
                                  use_sn=False,
                                  norm='bn',
                                  name='shared',
                                  name_offset=0)(input_layer)

        shared_conv = deform_conv_block(channels=self.config.deform_conv_channels[3:-1],
                                        deform_channels=self.config.deform_conv_offset_channels,
                                        kernel_size=self.config.kernel_size,
                                        strides=self.config.strides,
                                        activation=self.config.activation,
                                        name='shared',
                                        name_offset=2)(shared_conv)

        shared_conv = conv2d(shared_conv, filters=self.config.deform_conv_channels[-1], kernel_size=3, strides=2, use_sn=False,
                             norm='bn', activation='lrelu', name=f'shared_conv_{len(self.config.deform_conv_channels)-1}')
        conv_flatten = Flatten()(shared_conv)

        out = dense_branches(units=self.config.dense_branch_units,
                             output_num=self.config.num_classes,
                             activation=self.config.activation,
                             use_dropout=True,
                             dropout_rate=0.25,
                             name='class'
                             )(conv_flatten)

        return Model(input_layer, out, name='defromable_classifier')

    def build_model(self):
        model = self.define_model()
        parallel_model = multi_gpu_model(model, gpus=self.config.n_gpus) if self.config.n_gpus > 1 else model
        optimizer = Adam(self.config.lr, self.config.beta1, self.config.beta2, decay=0.01/30000)
        parallel_model.compile(optimizer=optimizer,
                               loss={"class_output": categorical_crossentropy},
                               metrics={"class_output": "accuracy"})
        return parallel_model
