import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import ReLU, LeakyReLU, Activation


def get_activation(activation):
    if activation is None or activation == 'linear':
        return lambda x: x
    elif activation == "relu":
        return ReLU()
    elif activation == "lrelu":
        return LeakyReLU(0.2)
    elif activation == "tanh":
        return Activation("tanh")
    elif activation == "sigmoid":
        return Activation("sigmoid")
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def normalize_img(img, label):
    normalized = tf.cast(img, tf.float32) / 127.5 - 1
    resized = tf.image.resize(normalized, (64, 64))
    return resized, label


def denormalize_img(img):
    return (((img*-1) + 1.) * 127.5).astype(np.uint8)


def get_dataloader(name, config):
    (train_ds, test_ds), ds_info = tfds.load(name,
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             data_dir=config.root_dir,
                                             as_supervised=True,
                                             with_info=True)

    config.input_shape = (64, 64, 3) #ds_info.features['image'].shape
    config.num_classes = ds_info.features['label'].num_classes
    config.num_steps = ds_info.splits['train'].num_examples // config.batch_size

    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
    train_ds = train_ds.batch(config.batch_size)
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=config.num_classes)))
    # https://www.tensorflow.org/guide/data_performance#prefetching
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # Caching is done after batching (as batches can be the same between epoch)
    test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(config.batch_size)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds

