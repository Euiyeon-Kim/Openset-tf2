import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions

from dataloader.imagenet_dataloader import DataLoader
from config import Config


IMAGE_SHAPE = (224, 224)
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"


if __name__ == '__main__':
    model = ResNet50(weights='imagenet')
    m = tf.keras.metrics.Accuracy()

    dataloader = DataLoader(Config, None)
    infer_dataloader = dataloader.get_infer_dataloaders()
    infer_dataloader = iter(infer_dataloader)

    accs = []
    for img, label in infer_dataloader:
        preds = model.predict(img)
        pred_label = np.argmax(preds, axis=-1)
        m.update_state(label, pred_label)
        acc = m.result().numpy()
        m.reset_states()
        print(acc)
        accs.append(acc)

    print(np.mean(accs))
