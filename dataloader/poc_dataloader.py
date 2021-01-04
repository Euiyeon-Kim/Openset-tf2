import os
from functools import partial

import numpy as np
import tensorflow as tf


def load_train_img_with_normalize(img_path, resize):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)         # output an RGB image
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, resize)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = np.random.randint(0, 3)
    img = tf.image.rot90(img, k)
    img = (img / 127.5) - 1
    return img


def load_test_img_with_normalize(img_path, resize):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)         # output an RGB image
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, resize)
    img = (img / 127.5) - 1
    return img


def load_train_img_fn(img_path, label, resize, num_classes):
    img = load_train_img_with_normalize(img_path, resize)
    label = tf.one_hot(label, depth=num_classes)
    return img, label


def load_test_img_fn(img_path, label, resize, num_classes):
    img = load_test_img_with_normalize(img_path, resize)
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
        self.val_infos = []
        self.test_infos = []
        self._set_infos()

    def _set_infos(self):
        with open(self._config.train_txt_path) as f:
            for line in f:
                self.train_infos.append(line)
            self.train_len = len(self.train_infos)

        with open(self._config.val_txt_path) as f:
            for line in f:
                self.val_infos.append(line)
            self.val_len = len(self.train_infos)

        with open(self._config.test_txt_path) as f:
            for line in f:
                self.test_infos.append(line)
            self.test_len = len(self.test_infos)

    def _train_data_generator(self, infos):
        while True:
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
            path = f'{self._config.root_dir}/{path}'
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
                                num_classes=self._config.num_classes)
        train_data_generator = partial(self._train_data_generator, infos=self.train_infos)
        train_ds = self._dataset_from_generator(train_data_generator, load_train_fn)

        load_val_fn = partial(load_test_img_fn,
                              resize=self._config.imagenet_resize,
                              num_classes=self._config.num_classes)
        val_data_generator = partial(self._test_data_generator, infos=self.val_infos)
        val_ds = self._dataset_from_generator(val_data_generator, load_val_fn, repeat=False, drop_remainder=False)
        return train_ds, val_ds

    def get_test_dataloaders(self):
        load_fn = partial(load_test_img_fn,
                          resize=self._config.imagenet_resize,
                          num_classes=self._config.num_classes)
        test_data_generator = partial(self._test_data_generator, infos=self.val_infos)
        test_ds = self._dataset_from_generator(test_data_generator, load_fn, repeat=False, drop_remainder=False)
        return test_ds

    def get_batch_size(self):
        if self._ms is None:
            batch_size = self._config.batch_size
        else:
            n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
            batch_size = self._config.batch_size * n_gpus
        return batch_size


# if __name__ == '__main__':
#     DISTINCT = ['n02129604', 'n04086273', 'n04254680', 'n07745940', 'n02690373', 'n03796401', 'n12620546', 'n11879895',
#                 'n02676566', 'n01806143', 'n02007558', 'n01695060', 'n03532672', 'n03065424', 'n03837869', 'n07711569',
#                 'n07734744', 'n03676483', 'n09229709', 'n07831146']
#     SIMILAR = ['n02100735', 'n02110185', 'n02096294', 'n02417914', 'n02110063', 'n02089867', 'n02102177', 'n02092339',
#                'n02098105', 'n02105641', 'n02096051', 'n02110341', 'n02086910', 'n02113712', 'n02113186', 'n02091467',
#                'n02106550', 'n02091831', 'n02104365', 'n02086079']
#
#     from glob import glob
#     val_paths = []
#     train_paths = []
#     test_paths = []
#     to_label_dict = {}
#     for idx, d in enumerate(SIMILAR):
#         to_label_dict[d] = idx
#         paths = glob(f'data/imagenet/train/{d}/*')
#         np.random.shuffle(paths)
#         val_paths.extend(paths[:50])
#         train_paths.extend(paths[50:])
#         paths = glob(f'data/imagenet/val/{d}/*')
#         np.random.shuffle(paths)
#         test_paths.extend(paths)
#
#     np.random.shuffle(train_paths)
#     np.random.shuffle(val_paths)
#     np.random.shuffle(test_paths)
#
#     with open('data/imagenet/poc_similar_test_infos.txt', 'w') as f:
#         for info in test_paths:
#             p = info[14:]
#             wnid = info[18:27]
#             label = to_label_dict[wnid]
#             f.write(f'{p} {label}\n')
#             print(p, wnid, label)
#
#     with open('data/imagenet/poc_similar_val_infos.txt', 'w') as f:
#         for info in val_paths:
#             p = info[14:]
#             wnid = info[20:29]
#             label = to_label_dict[wnid]
#             f.write(f'{p} {label}\n')
#             print(p, wnid, label)
#
#     with open('data/imagenet/poc_similar_train_infos.txt', 'w') as f:
#         for info in train_paths:
#             p = info[14:]
#             wnid = info[20:29]
#             label = to_label_dict[wnid]
#             f.write(f'{p} {label}\n')
#             print(p, wnid, label)
#

