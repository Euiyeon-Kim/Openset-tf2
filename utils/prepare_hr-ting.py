import os
import shutil
from glob import glob

import numpy as np
from scipy import io
from tqdm import tqdm


if __name__ == '__main__':
    np.random.seed(2017311385)
    meta = io.loadmat('data/meta.mat')
    synset_to_idx = {}
    original_idx_to_synset = {}
    synset_to_name = {}

    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
        synset = meta["synsets"][i, 0][1][0]
        name = meta["synsets"][i, 0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_idx[synset] = ilsvrc2012_id
        synset_to_name[synset] = name

    os.makedirs('data/hr-ting', exist_ok=True)
    paths = sorted(os.listdir('data/tiny-imagenet-200/train'))
    with open('data/hr-ting/hr-ting2imagenet_label_infos.txt', 'w') as f:
        for idx, path in enumerate(paths):
            info = f'{idx} {synset_to_idx[path]-1} {synset_to_name[path]}\n'
            f.write(info)

    paths = sorted(glob('data/tiny-imagenet-200/train/*/'))
    train_txt = open('data/hr-ting/hr-ting_train_infos.txt', 'w')
    val_txt = open('data/hr-ting/hr-ting_val_infos.txt', 'w')
    test_txt = open('data/hr-ting/hr-ting_test_infos.txt', 'w')

    for idx, path in tqdm(enumerate(paths)):
        wnid = path.split('/')[-2]
        hr_paths = glob(f'data/imagenet/train/{wnid}/*.JPEG')
        np.random.shuffle(hr_paths)

        os.makedirs(f'data/hr-ting/train/{wnid}', exist_ok=True)
        os.makedirs(f'data/hr-ting/val/{wnid}', exist_ok=True)
        os.makedirs(f'data/hr-ting/test', exist_ok=True)

        for p in hr_paths[:500]:
            info = f'train/{p[20:]} {idx}\n'
            train_txt.write(info)
            shutil.copyfile(p, f'data/hr-ting/train/{p[20:]}')

        for p in hr_paths[500:550]:
            info = f'val/{p[20:]} {idx}\n'
            val_txt.write(info)
            shutil.copyfile(p, f'data/hr-ting/val/{p[20:]}')

        for p in hr_paths[550:600]:
            info = f'test/{p[30:]} {idx}\n'
            test_txt.write(info)
            shutil.copyfile(p, f'data/hr-ting/test/{p[30:]}')

    train_txt.close()
    val_txt.close()
    test_txt.close()