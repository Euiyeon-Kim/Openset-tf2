from keras.layers import Conv2D, Dense, Add, Dropout

from models.modules.deformable import ConvOffset2D
from models.modules.spectral_norm import SN
from utils.utils import get_activation, get_normalization


def conv2d(x, filters, kernel_size=3, strides=1, use_sn=True, norm=None, activation=None, name=None):
    # Apply Spectral normalization
    x = SN(Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', name=name))(x) if use_sn else \
        Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', name=name)(x)

    # Normalization
    x = get_normalization(norm, name=f'{name}_norm')(x)

    # Activation
    x = get_activation(activation)(x)
    return x


def conv_blocks(channels, kernel_size, strides, activation, use_sn, norm, name, name_offset=0):
    def _conv_blocks(x):
        for idx, channel in enumerate(channels):
            x = conv2d(x, filters=channel, kernel_size=kernel_size, strides=strides,
                       use_sn=use_sn, norm=norm, activation=activation, name=f'{name}_conv_{idx+name_offset}')
        return x
    return _conv_blocks


def res_block(x, filters, kernel_size=3, strides=1, use_sn=True, norm=None, activation='lrelu', name=None):
    x = conv2d(x, filters=filters, kernel_size=3, strides=2, use_sn=False, norm='bn', activation='lrelu',
               name=f'{name}_shared_conv_0')
    _x = conv2d(x, filters, kernel_size=kernel_size, strides=strides, use_sn=use_sn, norm=norm, activation=activation,
                name=f'{name}_conv_1')
    _x = conv2d(_x, filters, kernel_size=kernel_size, strides=strides, use_sn=use_sn, norm=norm, activation=None,
                name=f'{name}_conv_2')
    return Add()([x, _x])


def res_blocks(channels, activation, use_sn, norm, name_offset=0):
    def _res_blocks(x):
        for idx, channel in enumerate(channels):
            x = res_block(x, filters=channel, kernel_size=3, strides=1,
                          use_sn=use_sn, norm=norm, activation=activation, name=f'res_{idx+name_offset}')
        return x
    return _res_blocks


def deform_conv_block(channels, deform_channels, kernel_size, activation, strides, name, name_offset):
    def _deform_conv_block(x):
        for idx, channel in enumerate(channels):
            x = ConvOffset2D(deform_channels[idx], name=f'{name}_deform_offset_{idx+name_offset}')(x)
            x = Conv2D(filters=channel, kernel_size=kernel_size, strides=strides, padding='same',
                       name=f'{name}_deform_conv_{idx+name_offset}')(x)
            x = get_activation(activation)(x)
        return x
    return _deform_conv_block


def dense_branches(units, output_num, activation, use_dropout, dropout_rate, name):
    def _dense_branches(x):
        for idx, unit in enumerate(units):
            x = Dense(units=unit, name=f'{name}_dense_{idx}')(x)
            x = get_activation(activation)(x)
            if use_dropout:
                x = Dropout(rate=dropout_rate)(x)
        x = Dense(units=output_num, activation='softmax', name=f'{name}_output')(x)
        return x
    return _dense_branches
