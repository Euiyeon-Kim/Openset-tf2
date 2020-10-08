from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from utils import normalize_img, resize_img


# Todo: Add openset selection process
def preprocess_dataset(ds, ds_info, config, train_data=True):
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if config.input_shape != ds_info.features['image'].shape:
        h, w, c = config.input_shape
        resize_function = partial(resize_img, size=(h, w))
        ds = ds.map(resize_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if train_data:
        ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(ds_info.splits['train'].num_examples)
        ds = ds.batch(config.batch_size)
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, depth=config.num_classes)))
    else:
        ds = ds.batch(config.batch_size)
        ds = ds.cache()

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_train_dataloader(name, config):
    (train_ds, test_ds), ds_info = tfds.load(name,
                                             split=['train', 'test'],
                                             shuffle_files=True,
                                             data_dir=config.tfds_dir,
                                             as_supervised=True,
                                             with_info=True)

    config.num_classes = ds_info.features['label'].num_classes
    config.num_steps = ds_info.splits['train'].num_examples // config.batch_size

    train_ds = preprocess_dataset(train_ds, ds_info, config, True)
    test_ds = preprocess_dataset(test_ds, ds_info, config, False)

    return train_ds, test_ds


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

