import os
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def load_img_with_normalize(img_path, resize, crop):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)         # output an RGB image
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, resize)
    img = tf.image.random_crop(img, crop)
    img = (img / 127.5) - 1
    return img


def load_img_fn(img_path, label, resize, crop, num_classes):
    img = load_img_with_normalize(img_path, resize, crop)
    label = tf.one_hot(label, depth=num_classes)
    return img, label


class DataLoader:
    def __init__(self, config, ms):
        self._config = config
        self._ms = ms
        self.train_len = 0

    def _data_generator(self, infos):
        while True:
            np.random.shuffle(infos)
            for info in infos:
                path, label = info.split(' ')
                path = f'{self._config.root_dir}/{path}'
                if os.path.isfile(path):
                    yield path, int(label)

    def get_train_dataloader(self):
        infos = []
        with open(self._config.train_txt_path) as f:
            for line in f:
                infos.append(line)
            self.train_len = len(infos)

        data_generator = partial(self._data_generator, infos=infos)
        load_fn = partial(load_img_fn, resize=self._config.imagenet_resize, crop=self._config.imagenet_crop, num_classes=self._config.num_classes)
        ds = tf.data.Dataset.from_generator(data_generator,
                                            output_types=(tf.string, tf.int32),
                                            output_shapes=((), ()))
        ds = ds.map(load_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.repeat()
        ds = ds.batch(self.get_batch_size(), drop_remainder=True)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if self._ms is not None:
            ds = self._ms.experimental_distribute_dataset(ds)
        return ds

    def get_batch_size(self):
        if self._ms is None:
            batch_size = self._config.batch_size
        else:
            n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
            batch_size = self._config.batch_size * n_gpus
        return batch_size



