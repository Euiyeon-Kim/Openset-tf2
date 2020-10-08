from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from utils import normalize_img, resize_img


def get_train_dataloader(name, config):
    [train_ds], ds_info = tfds.load(name,
                                    split=['train'],
                                    shuffle_files=True,
                                    data_dir=config.root_dir,
                                    as_supervised=True,
                                    with_info=True)

    config.num_classes = ds_info.features['label'].num_classes
    config.num_steps = ds_info.splits['train'].num_examples // config.batch_size

    # Normalization
    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Resize image
    if config.input_shape != ds_info.features['image'].shape:
        h, w, c = config.input_shape
        resize_function = partial(resize_img, size=(h, w))
        train_ds = train_ds.map(resize_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
    train_ds = train_ds.batch(config.batch_size)
    # train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, depth=config.num_classes)))
    # https://www.tensorflow.org/guide/data_performance#prefetching
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds


def get_test_dataloader(name, config):
    [test_ds], ds_info = tfds.load(name,
                                   split=['test'],
                                   shuffle_files=True,
                                   data_dir=config.root_dir,
                                   as_supervised=True,
                                   with_info=True)

    config.num_classes = ds_info.features['label'].num_classes

    # Normalization
    test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Resize image
    if config.input_shape != ds_info.features['image'].shape:
        h, w, c = config.input_shape
        resize_function = partial(resize_img, size=(h, w))
        test_ds = test_ds.map(resize_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Caching is done after batching (as batches can be the same between epoch)
    test_ds = test_ds.batch(config.batch_size)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return test_ds

