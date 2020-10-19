import numpy as np
from tqdm import tqdm

from config import Config, ModelStructure
from models.resnet50 import Resnet50
from models.deformable_classifier import DeformableClassifier
from dataloader.imagenet_dataloader import DataLoader
from utils.utils import openset_acc, closeset_acc, get_ms


def test_opened(classifier, dataloader):
    classifier.trainable = False
    openset_lengths = 0
    detected_as_openset_lengths = 0
    total_accs = []
    open_accs = []
    close_accs = []
    classifier.summary()
    for _, (img, label) in enumerate(dataloader):
        probs = classifier.predict(img)
        preds = np.argmax(probs, axis=-1)
        probs = np.max(probs, axis=-1)
        preds[probs < Config.threshold] = -1

        total_acc, open_acc, close_acc, openset_length, detected_as_openset_length = openset_acc(preds, label)

        total_accs.append(total_acc)
        open_accs.append(open_acc)
        close_accs.append(close_acc)
        openset_lengths += openset_length
        detected_as_openset_lengths += detected_as_openset_length

        print(f'Total: {total_acc} | Close: {close_acc} | Open: {open_acc}')

    print(f'\nClosed-set classification accuracy : {np.mean(close_accs)}')
    print(f'Open-set classification accuracy(Total: {openset_lengths}, Detected: {detected_as_openset_lengths}) : {np.mean(open_accs)}')
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
    from tensorflow import keras
    a = keras.applications.resnet.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
    a.summary()
    exit()

    # Label should be idx not one-hot vector
    ms = get_ms()
    dataloader = DataLoader(Config, ms)
    infer_ds = dataloader.get_test_dataloaders(include_openset=Config.test_with_openset)

    # Define model
    classifier = None
    if Config.structure == ModelStructure.RESNET50:
        classifier = Resnet50(Config).build_model()
    elif Config.structure == ModelStructure.DEFORM:
        classifier = DeformableClassifier(Config).build_model()
    else:
        raise Exception("Not implemented model structure")

    # Load model weight
    if not Config.classifier_weight_path:
        raise Exception('Need classifier weight path to load')
    classifier.load_weights(Config.classifier_weight_path)

    if Config.test_with_openset:
        test_opened(classifier, infer_ds)
    else:
        test_closed(classifier, infer_ds)


