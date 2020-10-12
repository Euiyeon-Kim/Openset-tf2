import json

import numpy as np
from matplotlib import pyplot as plt

from config import Config


if __name__ == '__main__':
    labels = []
    with open(Config.train_txt_path) as f:
        for line in f:
            labels.append(int(line.split(' ')[-1]))

    labels = np.array(labels)
    (unique, counts) = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    unique = unique[sorted_idx]
    counts = counts[sorted_idx]

    imagenet_name_json_path = 'data/imagenet.json'
    with open(imagenet_name_json_path) as json_file:
        json_data = json.load(json_file)

    wnid_infos = {}
    with open('data/words.txt') as f:
        for line in f:
            wnid = line[:9]
            word = line[10:-1]
            wnid_infos[word] = wnid

    imagenet_analysis_path = 'data/imagenet_train_info.txt'
    with open(imagenet_analysis_path, 'w') as f:
        for label, cnt in zip(unique, counts):
            str_labels = json_data[str(label)]
            info = f'{label:03}({wnid_infos[str_labels]}) {str_labels:125} {str(cnt)}\n'
            f.write(info)

    (cnt_unique, cnt_counts) = np.unique(counts, return_counts=True)
    plt.xlabel('Amount of data - acsending sort')
    plt.ylabel('# of classes witch has same amount of data')
    plt.bar(cnt_unique, cnt_counts)
    plt.savefig('data/imagenet_train_label_cnt.png')
