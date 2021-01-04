import os
import shutil
from termcolor import colored

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import multi_gpu_model

from config import Config
from utils.utils import get_ms
from models.modules.gradcam import GuidedGradCAM
from dataloader.poc_dataloader import DataLoader


def train(classifier, train_dataloader, val_dataloader):
    cam_dir = f'{Config.results_dir}/cam'
    log_dir = f'{Config.results_dir}/logs'
    chkpt_dir = f'{Config.results_dir}/chkpt'
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(log_dir, filename_suffix=f'_{str(Config.structure)}')
    loss_names = ['train/loss', 'train/acc']

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
                    _, val_acc = classifier.evaluate(val_img, val_label, verbose=0)
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
                classifier.save_weights(f"{chkpt_dir}/{str(Config.structure)}-{epoch+1}.h5")
            if os.path.isfile(f"{chkpt_dir}/{str(Config.structure)}-newest.h5"):
                os.remove(f"{chkpt_dir}/{str(Config.structure)}-newest.h5")
            classifier.save_weights(f"{chkpt_dir}/{str(Config.structure)}-newest.h5")


if __name__ == '__main__':
    os.makedirs(Config.results_dir, exist_ok=True)
    shutil.copyfile('config.py', f'{Config.results_dir}/config.py')

    ms = get_ms()

    dataloader = DataLoader(Config, ms)
    train_dataloader, val_dataloader = dataloader.get_train_dataloaders()
    train_dataloader = iter(train_dataloader)
    Config.num_steps = dataloader.train_len // dataloader.get_batch_size()

    with ms.scope():
        model = tf.keras.applications.EfficientNetB0(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=20,
            classifier_activation="softmax",
        )
        parallel_model = multi_gpu_model(model, gpus=Config.n_gpus) if Config.n_gpus > 1 else model
        optimizer = Adam(Config.lr, Config.beta1, Config.beta2, decay=0.01 / 30000)
        model.compile(optimizer=optimizer,
                      loss={"predictions": categorical_crossentropy},
                      metrics={"predictions": "accuracy"})
        model.summary()

    train(model, train_dataloader, val_dataloader)
