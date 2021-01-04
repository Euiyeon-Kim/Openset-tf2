import os
from glob import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_threshold_mask(h, w, r):
    ch, cw = int(h / 2), int(w / 2)
    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if np.sqrt((i - ch) ** 2 + (j - cw)**2) < r:
                mask[i, j] = 1.0
    return mask


def fourier_transofrmation(img, r, mode='low'):
    h, w, c = img.shape
    mask = get_threshold_mask(h, w, r)
    spectrum = np.zeros_like(img)
    lfc = np.zeros_like(img)
    hfc = np.zeros_like(img)

    for i in range(c):
        origin = img[:, :, i]
        f = np.fft.fft2(origin)
        f_shift = np.fft.fftshift(f)
        spectrum[:, :, i] = 20 * np.log(np.abs(f_shift))

        low_masked = np.multiply(f_shift, mask)
        low_f_ishift = np.fft.ifftshift(low_masked)
        lfc[:, :, i] = np.abs(np.fft.ifft2(low_f_ishift))

        high_masked = f_shift * (1 - mask)
        high_f_ishift = np.fft.ifftshift(high_masked)
        hfc[:, :, i] = np.abs(np.fft.ifft2(high_f_ishift))
        # f_ishift = np.fft.ifftshift(f_shift)
        # recover[:, :, i] = np.real(np.fft.ifft2(f_ishift))

    if mode == 'lfc':
        return lfc
    elif mode == 'hfc':
        return hfc
    else:
        return spectrum, lfc, hfc


if __name__ == '__main__':
    classes = (os.listdir('../data/imagenet/train'))[13:]

    for wnid in classes:
        paths = glob(f'../data/imagenet/train/{wnid}/*')[:20]
        lfcs = np.zeros(256)
        hfcs = np.zeros(256)
        for path in paths:
            img = cv2.imread(path)
            spec, lfc, hfc = fourier_transofrmation(img, r=16, mode='both')
            cv2.imshow("spec", np.uint8(spec))
            cv2.imshow("origin", img)
            cv2.imshow("l", lfc)
            cv2.imshow("h", hfc)
            cv2.waitKey()

        exit()

