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


def fourier_transofrmation(img, r):
    h, w, c = img.shape
    mask = get_threshold_mask(h, w, r)
    recover = np.zeros_like(img)
    lfc = np.zeros_like(img)
    hfc = np.zeros_like(img)
    for i in range(c):
        origin = img[:, :, i]

        # Fourier transformation
        f = np.fft.fft2(origin)
        f_shift = np.fft.fftshift(f)

        # Thresholding
        low_masked = np.multiply(f_shift, mask)
        high_masked = f_shift * (1 - mask)

        f_ishift = np.fft.ifftshift(f_shift)
        low_f_ishift = np.fft.ifftshift(low_masked)
        high_f_ishift = np.fft.ifftshift(high_masked)

        recover[:, :, i] = np.real(np.fft.ifft2(f_ishift))
        lfc[:, :, i] = np.abs(np.fft.ifft2(low_f_ishift))
        hfc[:, :, i] = np.abs(np.fft.ifft2(high_f_ishift))

    cv2.imshow('l', np.uint8(lfc))
    cv2.imshow("h", np.uint8(hfc))
    cv2.imshow("r", np.uint8(recover))
    cv2.waitKey()

    return lfc, hfc


if __name__ == '__main__':
    img = cv2.imread('../test.JPEG')
    fourier_transofrmation(img, 32)
