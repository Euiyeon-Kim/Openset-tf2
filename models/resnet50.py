from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, Dense
from tensorflow_addons.layers import InstanceNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import multi_gpu_model


class Resnet50:
    def __init__(self, config):
        self.config = config

    def define_model(self):
        input_layer = Input(shape=self.config.input_shape)

        x = ZeroPadding2D(padding=(3, 3))(input_layer)
        x = Conv2D(64, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

        x = MaxPooling2D((3, 3), 2)(x)

        shortcut = x
        for i in range(3):
            if i == 0:
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
                x = InstanceNormalization()(x)
                shortcut = InstanceNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

        shortcut = x
        for i in range(4):
            if i == 0:
                x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = InstanceNormalization()(x)
                shortcut = InstanceNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

        shortcut = x
        for i in range(6):
            if i == 0:
                x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = InstanceNormalization()(x)
                shortcut = InstanceNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

        shortcut = x
        for i in range(3):
            if i == 0:
                x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = InstanceNormalization()(x)
                shortcut = InstanceNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
                x = InstanceNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
                x = InstanceNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

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
