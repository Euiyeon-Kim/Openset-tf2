import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from utils import denormalize_img


class GuidedGradCAM:
    def __init__(self, classifier, conv_layer_name, eps=1e-8):
        self.model = Model(inputs=classifier.inputs,
                           outputs=[classifier.get_layer(conv_layer_name).output, classifier.output])
        self.eps = eps

    def generate(self, img, class_idx):
        _, h, w, _ = img.shape
        with tf.GradientTape() as tape:
            conv_output, predictions = self.model(img)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_output)

        relu_conv_output = tf.cast(conv_output > 0, "float32")
        relu_grads = tf.cast(grads > 0, "float32")
        guided_grads = relu_conv_output * relu_grads * grads

        conv_output = conv_output[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1)




def gradCAM(model, visualize_layer, img, class_idx, eps=1e-8):
    _, h, w, _ = img.shape
    cam_model = Model(model.inputs, [model.get_layer(visualize_layer).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = cam_model(img)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)

    castConvOutputs = tf.cast(conv_outputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads

    convOutputs = conv_outputs[0]
    guidedGrads = guidedGrads[0]

    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    heatmap = cv2.resize(cam.numpy(), (w, h))
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    sample_img = denormalize_img(img)[0].astype('uint8')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cam_img = cv2.addWeighted(sample_img, 0.9, heatmap, 1, 0)

    return sample_img, cam_img
