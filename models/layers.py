from keras.layers import Conv2D, Dense

from utils import get_activation


# Vanilla classifier
def conv_blocks(channels, kernel_size, strides, activation, name):
    def _conv_blocks(x):
        for idx, channel in enumerate(channels):
            x = Conv2D(filters=channel, kernel_size=kernel_size, strides=strides, padding='same', name=f'{name}_conv_{idx}')(x)
            x = get_activation(activation)(x)
        return x
    return _conv_blocks


def dense_branches(units, output_num, activation, name):
    def _dense_branches(x):
        for idx, unit in enumerate(units):
            x = Dense(units=unit, name=f'{name}_dense_{idx}')(x)
            x = get_activation(activation)(x)
        x = Dense(units=output_num, activation='softmax', name=f'{name}_output')(x)
        return x
    return _dense_branches

