import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import Config, ModelStructure
from models.modules.gradcam import GuidedGradCAM
from models.vanilla_classifier import VanillaClassifier
from dataloader.tfds import get_train_dataloader


def train(classifier, dataloader):
    cam_dir = f'{Config.results_dir}/cam'
    log_dir = f'{Config.results_dir}/logs'
    chkpt_dir = f'{Config.results_dir}/chkpt'
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    writer = tf.summary.create_file_writer(log_dir, filename_suffix=f'_{str(Config.structure)}')
    loss_names = ["ce_loss", "acc"]

    classifier.summary()
    with writer.as_default():

        for epoch in tqdm(range(Config.num_epochs), desc="Training Classifier"):
            for step, (img, label) in enumerate(dataloader):
                losses = classifier.train_on_batch(img, label)

                # Log losses
                for loss_name, loss in zip(loss_names, losses):
                    tf.summary.scalar(loss_name, loss, step=epoch*Config.num_steps+step)
                writer.flush()

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
    train_ds = get_train_dataloader(Config.dataset_name, Config)

    # Define model
    classifier = None
    if Config.structure == ModelStructure.VANILLA:
        classifier = VanillaClassifier(Config).build_model()
    # elif Config.structure == ModelStructure.DEFORM:
    # elif Config.structure == ModelStructure.SENET:
    # elif Config.structure == ModelStructure.SPNORM:
    # elif Config.structure == ModelStructure.CBAM:
    # else:
    #     raise Exception("Not implemented model structure")

    if Config.classifier_weight_path:
        classifier.load_weights(Config.classifier_weight_path)

    train(classifier, train_ds)
