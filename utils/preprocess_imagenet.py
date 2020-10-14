import os
import json
import shutil
from glob import glob

import numpy as np
from scipy import io
from matplotlib import pyplot as plt

from config import Config


if __name__ == '__main__':
    meta = io.loadmat('data/meta.mat')
    original_idx_to_synset = {}
    synset_to_name = {}
    synset_to_idx = {}
    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
        synset = meta["synsets"][i, 0][1][0]
        name = meta["synsets"][i, 0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name
        synset_to_idx[synset] = ilsvrc2012_id

    # Save train dataset infos
    train_wnids = sorted(os.listdir('data/imagenet/train'))
    with open('data/imagenet/imagenet2012_train_label_infos.txt', 'w') as f:
        for i in range(1000):
            wnid = original_idx_to_synset[i+1]
            name = synset_to_name[wnid]
            info = f'{i:03} {wnid} {name}\n'
            f.write(info)

    train_paths = sorted(glob('data/imagenet/train/*/*.JPEG'))
    with open('data/imagenet/imagenet2012_train_image_infos.txt', 'w') as f:
        for path in train_paths:
            p = path[14:]
            wnid = path[20:29]
            label = synset_to_idx[wnid] - 1
            info = f'{p} {label}\n'
            f.write(info)

    # Preprocess validation dataset
    val_paths = sorted(glob('data/imagenet/val/*.JPEG'))
    labels = []
    with open('data/ILSVRC2012_validation_ground_truth.txt') as f:
        for line in f:
            labels.append(int(line.split(' ')[-1]))
    labels = labels[4:]

    for path, label in zip(val_paths, labels):
        wnid = original_idx_to_synset[label]
        name = synset_to_name[wnid]
        confirm = os.path.isdir(f'data/imagenet/train/{wnid}')
        os.makedirs(f'data/imagenet/val/{wnid}', exist_ok=True)
        shutil.move(path, f'data/imagenet/val/{wnid}')

    # labels = []
    # with open(Config.train_txt_path) as f:
    #     for line in f:
    #         labels.append(int(line.split(' ')[-1]))
    #
    # labels = np.array(labels)
    # (unique, counts) = np.unique(labels, return_counts=True)
    # sorted_idx = np.argsort(counts)[::-1]
    # unique = unique[sorted_idx]
    # counts = counts[sorted_idx]

    # imagenet_name_json_path = 'data/imagenet.json'
    # with open(imagenet_name_json_path) as json_file:
    #     json_data = json.load(json_file)

    # name_to_wnid = {}
    # with open('data/words.txt') as f:
    #     for line in f:
    #         wnid = line[:9]
    #         word = line[10:-1]
    #         name_to_wnid[word] = wnid


    # imagenet_analysis_path = 'data/imagenet_train_info.txt'
    # with open(imagenet_analysis_path, 'w') as f:
    #     for label, cnt in zip(unique, counts):
    #         str_labels = json_data[str(label)]
    #         info = f'{label:03}({wnid_infos[str_labels]}) {str_labels:125} {str(cnt)}\n'
    #         f.write(info)
    #
    # (cnt_unique, cnt_counts) = np.unique(counts, return_counts=True)
    # plt.xlabel('Amount of data - acsending sort')
    # plt.ylabel('# of classes witch has same amount of data')
    # plt.bar(cnt_unique, cnt_counts)
    # plt.savefig('data/imagenet_train_label_cnt.png')
