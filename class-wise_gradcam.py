import os
import argparse
from glob import glob

import cv2

from config import Config, ModelStructure
from models.vanilla_classifier import VanillaClassifier
from models.deformable_classifier import DeformableClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wnid', type=str, default='n01494475', help='class wordnet id')
    args = parser.parse_args()

    if Config.structure == ModelStructure.VANILLA:
        classifier = VanillaClassifier(Config).build_model()
    elif Config.structure == ModelStructure.DEFORM:
        classifier = DeformableClassifier(Config).build_model()
    else:
        raise Exception("Not implemented model structure")

    if not Config.classifier_weight_path:
        raise Exception('Need classifier weight path to load')
    classifier.load_weights(Config.classifier_weight_path)

    paths = glob(f'{Config.root_dir}/train/{args.wnid}/*.JPEG')
    for path in paths:
        print(cv2.imread(path).shape)
        exit()

