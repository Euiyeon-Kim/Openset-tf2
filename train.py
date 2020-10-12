import os
import shutil
from termcolor import colored

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import Config, ModelStructure
from models.modules.gradcam import GuidedGradCAM
from models.vanilla_classifier import VanillaClassifier
from models.deformable_classifier import DeformableClassifier
from models.sn_classifier import SpecNormClassifier
from dataloader.tfds import get_train_dataloader
from dataloader.imagenet_dataloader import DataLoader
from utils.utils import get_ms, closeset_acc


def train(classifier, train_dataloader, val_dataloader):
    cam_dir = f'{Config.results_dir}/cam'
    log_dir = f'{Config.results_dir}/logs'
    chkpt_dir = f'{Config.results_dir}/chkpt'
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(log_dir, filename_suffix=f'_{str(Config.structure)}')
    loss_names = ["train/ce_loss", "train/acc"]

    best_val_acc = 0
    print(colored(f'Start training [{Config.results_dir}]', 'green'))
    with writer.as_default():
        for epoch in range(Config.num_epochs):
            for step in tqdm(range(Config.num_steps), desc=f"Epoch {epoch}"):
                # Train
                img, label = next(train_dataloader)
                losses = classifier.train_on_batch(img, label)

                # Log losses
                for loss_name, loss in zip(loss_names, losses):
                    tf.summary.scalar(loss_name, loss, step=epoch * Config.num_steps + step)
                writer.flush()
            print(losses)

            # Validation
            if (epoch + 1) % Config.epochs_to_validate == 0:
                val_accs = []
                for val_img, val_label in val_dataloader:
                    val_probs = classifier.predict(val_img)
                    val_preds = np.argmax(val_probs, axis=-1)
                    val_probs = np.max(val_probs, axis=-1)
                    val_preds[val_probs < Config.threshold] = -1
                    val_acc = closeset_acc(val_preds, val_label)
                    val_accs.append(val_acc)

                tf.summary.scalar('val_acc', np.mean(val_accs), step=epoch)

                if np.mean(val_accs) > best_val_acc:
                    best_val_acc = np.mean(val_accs)
                    classifier.save_weights(f"{chkpt_dir}/{str(Config.structure)}-classifier-best.h5")

            # Save GradCAM
            if (epoch + 1) % Config.epochs_to_save_gradCAM == 0:
                cam_model = GuidedGradCAM(classifier, Config.cam_layer)
                preds = classifier.predict_on_batch(img)
                predicted_class = preds.argmax(axis=1)[0]
                real_class = np.argmax(label.numpy(), axis=1)[0]
                sample_img, cam_img = cam_model.generate(np.expand_dims(img[0], axis=0), predicted_class)

                cv2.imwrite(f'{cam_dir}/{epoch}_{predicted_class}_{real_class}_sample.png', sample_img)
                cv2.imwrite(f'{cam_dir}/{epoch}_{predicted_class}_{real_class}_cam.png', cam_img)

                del cam_model

            # Save weights
            if (epoch+1) % Config.epochs_to_save_weights == 0 or (epoch+1) == Config.num_epochs:
                classifier.save_weights(f"{chkpt_dir}/{str(Config.structure)}-classifier-{epoch+1}.h5")


if __name__ == '__main__':
    os.makedirs(Config.results_dir, exist_ok=True)
    shutil.copyfile('config.py', f'{Config.results_dir}/config.py')
    # Load dataloaders
    if Config.use_tfds:
        train_dataloader, val_dataloader = get_train_dataloader(Config.dataset_name, Config)
        train_dataloader = iter(train_dataloader)
        # val_dataloader = iter(val_dataloader)
    else:
        ms = get_ms()
        dataloader = DataLoader(Config, ms)
        train_dataloader, val_dataloader = dataloader.get_train_dataloaders()
        train_dataloader = iter(train_dataloader)
        Config.num_steps = dataloader.train_len // dataloader.get_batch_size()

    # Define model
    classifier = None
    if Config.structure == ModelStructure.VANILLA:
        classifier = VanillaClassifier(Config).build_model()
    elif Config.structure == ModelStructure.DEFORM:
        classifier = DeformableClassifier(Config).build_model()
    elif Config.structure == ModelStructure.SPNORM:
        classifier = SpecNormClassifier(Config).build_model()
    else:
        raise Exception("Not implemented model structure")
    classifier.summary()

    if Config.classifier_weight_path:
        classifier.load_weights(Config.classifier_weight_path)
        print(colored(f'Loaded classifier weight from [{Config.classifier_weight_path}]', 'green'))

    train(classifier, train_dataloader, val_dataloader)
    # test_opened(classifier, test_dataloader)
