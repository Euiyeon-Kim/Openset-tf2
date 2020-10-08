import numpy as np
from tqdm import tqdm

from config import Config, ModelStructure
from models.vanilla_classifier import VanillaClassifier
from dataloader.tfds import get_infer_dataloader
from utils import openset_acc, closeset_acc


def test_opened(classifier, dataloader):
    classifier.trainable = False
    openset_lengths = 0
    total_accs = []
    open_accs = []
    close_accs = []
    for _, (img, label) in tqdm(enumerate(dataloader)):
        probs = classifier.predict(img)
        preds = np.argmax(probs, axis=-1)
        probs = np.max(probs, axis=-1)
        preds[probs < Config.threshold] = -1
        total_acc, open_acc, close_acc, openset_length = openset_acc(preds, label)
        total_accs.append(total_acc)
        open_accs.append(open_acc)
        close_accs.append(close_acc)
        openset_lengths += openset_length

    print(f'Closed-set classification accuracy : {np.mean(close_accs)}')
    print(f'Open-set classification accuracy(Total length: {openset_lengths}) : {np.mean(open_accs)}')
    print(f'Total classification accuracy : {np.mean(total_accs)}')


def test_closed(classifier, dataloader):
    classifier.trainable = False
    total_accs = []
    for _, (img, label) in tqdm(enumerate(dataloader)):
        probs = classifier.predict(img)
        preds = np.argmax(probs, axis=-1)
        acc = closeset_acc(preds, label)
        total_accs.append(acc)
    print(f'Total classification accuracy : {np.mean(total_accs)}')


if __name__ == '__main__':
    # Label should be idx not one-hot vector
    infer_ds = get_infer_dataloader(Config.dataset_name, Config)

    # Define model
    classifier = None
    if Config.structure == ModelStructure.VANILLA:
        classifier = VanillaClassifier(Config).build_model()

    # Load model weight
    if not Config.classifier_weight_path:
        raise Exception('Need classifier weight path to load')
    classifier.load_weights(Config.classifier_weight_path)

    test_opened(classifier, infer_ds)


