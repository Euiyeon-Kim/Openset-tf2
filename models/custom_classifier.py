from keras.models import Model
from keras.layers import Input, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import multi_gpu_model

from models.layers import dense_branches, res_blocks


class CustomClassifier:
    def __init__(self, config):
        self.config = config

    def define_model(self):
        img = Input(shape=self.config.input_shape)

        shared_conv = res_blocks(channels=self.config.shared_conv_channels,
                                 activation=self.config.activation,
                                 use_sn=True,
                                 norm=None)(img)

        conv_flatten = Flatten()(shared_conv)

        out = dense_branches(units=self.config.dense_branch_units,
                             output_num=self.config.num_classes,
                             activation=self.config.activation,
                             use_dropout=True,
                             dropout_rate=0.25,
                             name='class'
                             )(conv_flatten)

        return Model(img, out, name='custom_classifier')

    def build_model(self):
        model = self.define_model()
        parallel_model = multi_gpu_model(model, gpus=self.config.n_gpus) if self.config.n_gpus > 1 else model
        optimizer = Adam(self.config.lr, self.config.beta1, self.config.beta2, decay=0.01/30000)
        parallel_model.compile(optimizer=optimizer,
                               loss={"class_output": categorical_crossentropy},
                               metrics={"class_output": "accuracy"})
        return parallel_model
