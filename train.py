import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import Config, ModelStructure
from models.gradcam import gradCAM
from models.vanilla_classifier import VanillaClassifier
from utils import get_dataloader, denormalize_img


def train(classifier, dataloader, cam_layer):
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
                preds = classifier.predict_on_batch(img)
                predicted_class = preds.argmax(axis=1)[0]
                real_class = np.argmax(label.numpy(), axis=1)[0]
                sample_img, cam_img = gradCAM(classifier, cam_layer, np.expand_dims(img[0], axis=0), predicted_class)

                cv2.imwrite(f'{cam_dir}/{epoch}_{predicted_class}_{real_class}_sample.png', sample_img)
                cv2.imwrite(f'{cam_dir}/{epoch}_{predicted_class}_{real_class}_cam.png', cam_img)

            # Save weights
            if (epoch+1) % Config.epochs_to_save_weights == 0 or (epoch+1) == Config.num_epochs:
                save_path = f"{chkpt_dir}/classifier-{epoch+1}.ckpt"
                classifier.save_weights(save_path)


if __name__ == '__main__':
    train_ds, test_ds = get_dataloader('svhn_cropped', Config)

    # Define model
    classifier = None
    if Config.structure == ModelStructure.VANILLA:
        classifier = VanillaClassifier(Config).build_model()
    # elif Config.structure == ModelStructure.DEFORM:
    # elif Config.structure == ModelStructure.SENET:
    # elif Config.structure == ModelStructure.SPNORM:
    # else:
    #     raise Exception("Not implemented model structure")

    train(classifier, train_ds, "shared_conv_3")