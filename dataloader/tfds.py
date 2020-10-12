from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from utils.utils import normalize_img, resize_img


def filter_openset_train(_, label, openset_labels):
    is_allowed = tf.equal(openset_labels, label)
    reduced = tf.reduce_sum(tf.cast(is_allowed, tf.float32))
    return tf.logical_not(tf.greater(reduced, tf.constant(0.)))


def filter_openset_test(img, label, openset_labels):
    is_allowed = tf.equal(openset_labels, label)
    reduced = tf.reduce_sum(tf.cast(is_allowed, tf.float32))
    is_openset = tf.greater(reduced, tf.constant(0.))

    def open_label():
        return tf.convert_to_tensor(-1, dtype=tf.int64)

    def allowed_label():
        return label

    label = tf.cond(is_openset, open_label, allowed_label)
    return img, label


def preprocess_dataset(ds, ds_info, config, train_step=True):
    if train_step:
        filter_fn = partial(filter_openset_train, openset_labels=config.openset_labels)
        ds = ds.filter(filter_fn)
    else:
        filter_fn = partial(filter_openset_test, openset_labels=config.openset_labels)
        ds = ds.map(filter_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if config.input_shape != ds_info.features['image'].shape:
        h, w, c = config.input_shape
        resize_function = partial(resize_img, size=(h, w))
        ds = ds.map(resize_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if train_step:
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=config.num_classes)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(ds_info.splits['train'].num_examples)
        ds = ds.batch(config.batch_size)

    else:
        ds = ds.batch(config.batch_size)
        ds = ds.cache()

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_train_dataloader(name, config):
    splits = tfds.Split.TRAIN.subsplit(weighted=config.split_weight)
    (train_ds, val_ds), ds_info = tfds.load(name,
                                            split=splits,
                                            shuffle_files=True,
                                            data_dir=config.tfds_dir,
                                            as_supervised=True,
                                            with_info=True)

    config.num_classes = ds_info.features['label'].num_classes
    config.num_steps = ds_info.splits['train'].num_examples // config.batch_size

    openset_labels = np.random.choice(config.num_classes, int(config.num_classes*config.openset_rate), replace=False)
    f = open(f'{config.results_dir}/openset_labels.txt', 'w')
    np.savetxt(f, openset_labels.astype(int), fmt='%i')
    f.close()
    config.openset_labels = tf.convert_to_tensor(openset_labels, tf.int64)

    train_ds = preprocess_dataset(train_ds, ds_info, config, True)
    val_ds = preprocess_dataset(val_ds, ds_info, config, False)

    return train_ds, val_ds


def get_infer_dataloader(name, config):
    [test_ds], ds_info = tfds.load(name,
                                   split=['test'],
                                   shuffle_files=True,
                                   data_dir=config.tfds_dir,
                                   as_supervised=True,
                                   with_info=True)

    config.num_classes = ds_info.features['label'].num_classes
    preprocess_dataset(test_ds, ds_info, config, False)
    return test_ds

