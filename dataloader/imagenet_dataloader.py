import os
from functools import partial

import numpy as np
import tensorflow as tf


def load_train_img_with_normalize(img_path, resize, crop):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)         # output an RGB image
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, resize)
    img = tf.image.random_crop(img, crop)
    img = (img / 127.5) - 1
    return img


def load_test_img_with_normalize(img_path, ratio):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)         # output an RGB image
    img = tf.cast(img, tf.float32)
    img = tf.image.central_crop(img, ratio)
    img = (img / 127.5) - 1
    return img


def load_train_img_fn(img_path, label, resize, crop, num_classes):
    img = load_train_img_with_normalize(img_path, resize, crop)
    label = tf.one_hot(label, depth=num_classes)
    return img, label


def load_test_img_fn(img_path, label, resize, crop, num_classes):
    img = load_train_img_with_normalize(img_path, resize, crop)
    label = tf.one_hot(label, depth=num_classes)
    return img, label


class DataLoader:
    def __init__(self, config, ms):
        self._config = config
        self._ms = ms
        self.train_len = 0
        self.val_len = 0
        self.test_len = 0
        self.train_infos = []
        self.closed_test_len = 0
        self.opend_test_len = 0

    def _train_data_generator(self, infos):
        while True:
            np.random.shuffle(infos)
            for info in infos:
                path, label = info.split(' ')
                path = f'{self._config.root_dir}/{path}'
                if os.path.isfile(path):
                    yield path, int(label)

    def _val_data_generator(self, infos):
        np.random.shuffle(infos)
        for info in infos:
            path, label = info.split(' ')
            path = f'{self._config.root_dir}/{path}'
            if os.path.isfile(path):
                yield path, int(label)

    def _test_data_generator(self, infos):
        np.random.shuffle(infos)
        for info in infos:
            path, label = info.split(' ')
            if path[0] == 'v':
                path = f'{self._config.root_dir}/val/{path.split("/")[-1]}'
            else:
                path = f'{self._config.openset_dir}/{path}'
            if os.path.isfile(path):
                yield path, int(label)

    def _dataset_from_generator(self, data_generator, load_fn, repeat=True, drop_remainder=True):
        ds = tf.data.Dataset.from_generator(data_generator,
                                            output_types=(tf.string, tf.int32),
                                            output_shapes=((), ()))
        ds = ds.map(load_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(self.get_batch_size(), drop_remainder=drop_remainder)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if self._ms is not None:
            ds = self._ms.experimental_distribute_dataset(ds)
        return ds

    def get_train_dataloaders(self):
        load_train_fn = partial(load_train_img_fn,
                                resize=self._config.imagenet_resize,
                                crop=self._config.imagenet_crop,
                                num_classes=self._config.num_classes)

        train_infos = []
        with open(self._config.train_txt_path) as f:
            for line in f:
                train_infos.append(line)
            self.train_len = len(train_infos)
        train_data_generator = partial(self._train_data_generator, infos=train_infos)
        train_ds = self._dataset_from_generator(train_data_generator, load_train_fn)

        load_val_fn = partial(load_test_img_fn,
                              resize=self._config.imagenet_resize,
                              crop=self._config.imagenet_crop,
                              num_classes=self._config.num_classes)
        val_infos = []
        with open(self._config.val_txt_path) as f:
            for line in f:
                val_infos.append(line)
            self.val_len = len(val_infos)
        val_data_generator = partial(self._val_data_generator, infos=val_infos)
        val_ds = self._dataset_from_generator(val_data_generator, load_val_fn, repeat=False, drop_remainder=False)
        return train_ds, val_ds

    def get_test_dataloaders(self, include_openset=True):
        load_fn = partial(load_test_img_fn,
                          crop=self._config.imagenet_crop)

        close_infos = []
        with open(self._config.test_txt_path) as f:
            for line in f:
                close_infos.append(line)
            self.closed_test_len = len(close_infos)

        if include_openset:
            open_infos = []
            with open(self._config.openset_txt_path) as f:
                for line in f:
                    open_infos.append(line)
                self.opend_test_len = len(open_infos)
            test_infos = open_infos + close_infos
            test_data_generator = partial(self._test_data_generator, infos=test_infos)
        else:
            test_data_generator = partial(self._test_data_generator, infos=close_infos)

        test_ds = self._dataset_from_generator(test_data_generator, load_fn, repeat=False, drop_remainder=False)
        return test_ds

    def get_batch_size(self):
        if self._ms is None:
            batch_size = self._config.batch_size
        else:
            n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
            batch_size = self._config.batch_size * n_gpus
        return batch_size



