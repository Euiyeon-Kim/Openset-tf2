from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, Dense
from tensorflow_addons.layers import InstanceNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import multi_gpu_model

from models.layers import conv2d
from utils.utils import get_activation


class Resnet50:
    def __init__(self, config):
        self.config = config

    def define_model(self):
        input_layer = Input(shape=self.config.input_shape)

        x = conv2d(input_layer, filters=64, kernel_size=7, strides=2,
                   pad_type='zero', pad_size=3, norm='in', activation='lrelu')
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), 2)(x)

        shortcut = conv2d(x, filters=256, kernel_size=1, strides=1, pad_type='valid', norm='in')
        for i in range(3):
            x = conv2d(x, filters=64, kernel_size=1, strides=1, pad_type='valid', norm='in', activation='lrelu')
            x = conv2d(x, filters=64, kernel_size=3, strides=1, pad_type='same', norm='in', activation='lrelu')
            x = conv2d(x, filters=256, kernel_size=1, strides=1, pad_type='valid', norm='in')

            x = Add()([x, shortcut])
            x = get_activation('lrelu')(x)
            shortcut = x

        shortcut = conv2d(x, filters=512, kernel_size=1, strides=2, pad_type='valid', norm='in')
        for i in range(4):
            x = conv2d(x, filters=128, kernel_size=1, strides=2, pad_type='valid', norm='in', activation='lrelu') if i == 0 \
                else conv2d(x, filters=128, kernel_size=1, strides=1, pad_type='valid', norm='in', activation='lrelu')
            x = conv2d(x, filters=128, kernel_size=3, strides=1, pad_type='same', norm='in', activation='lrelu')
            x = conv2d(x, filters=512, kernel_size=1, strides=1, pad_type='valid', norm='in')

            x = Add()([x, shortcut])
            x = get_activation('lrelu')(x)
            shortcut = x

        shortcut = conv2d(x, filters=1024, kernel_size=1, strides=2, pad_type='valid', norm='in')
        for i in range(6):
            x = conv2d(x, filters=256, kernel_size=1, strides=2, pad_type='valid', norm='in', activation='lrelu') if i == 0 \
                else conv2d(x, filters=256, kernel_size=1, strides=1, pad_type='valid', norm='in', activation='lrelu')
            x = conv2d(x, filters=256, kernel_size=3, strides=1, pad_type='same', norm='in', activation='lrelu')
            x = conv2d(x, filters=1024, kernel_size=1, strides=1, pad_type='valid', norm='in')

            x = Add()([x, shortcut])
            x = get_activation('lrelu')(x)
            shortcut = x

        shortcut = conv2d(x, filters=2048, kernel_size=1, strides=2, pad_type='valid', norm='in')
        for i in range(3):
            x = conv2d(x, filters=512, kernel_size=1, strides=2, pad_type='valid', norm='in', activation='lrelu') if i == 0 \
                else conv2d(x, filters=512, kernel_size=1, strides=1, pad_type='valid', norm='in', activation='lrelu')
            x = conv2d(x, filters=512, kernel_size=3, strides=1, pad_type='same', norm='in', activation='lrelu')
            x = conv2d(x, filters=2048, kernel_size=1, strides=1, pad_type='valid', norm='in')

            x = Add()([x, shortcut])
            x = get_activation('lrelu')(x)
            shortcut = x

        x = GlobalAveragePooling2D()(x)
        out = Dense(self.config.num_classes, activation='softmax', name=f'class_output')(x)
        return Model(inputs=input_layer, outputs=out, name='resnet50')

    def build_model(self):
        model = self.define_model()
        parallel_model = multi_gpu_model(model, gpus=self.config.n_gpus) if self.config.n_gpus > 1 else model
        optimizer = Adam(self.config.lr, self.config.beta1, self.config.beta2, decay=0.01/30000)
        parallel_model.compile(optimizer=optimizer,
                               loss={"class_output": categorical_crossentropy},
                               metrics={"class_output": "accuracy"})
        return parallel_model
