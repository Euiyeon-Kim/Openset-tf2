import os
from glob import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt

DISTINCT = ['n02129604', 'n04086273', 'n04254680', 'n07745940', 'n02690373', 'n03796401', 'n12620546', 'n11879895',
            'n02676566', 'n01806143', 'n02007558', 'n01695060', 'n03532672', 'n03065424', 'n03837869', 'n07711569',
            'n07734744', 'n03676483', 'n09229709', 'n07831146']
SIMILAR = ['n02100735', 'n02110185', 'n02096294', 'n02417914', 'n02110063', 'n02089867', 'n02102177', 'n02092339',
           'n02098105', 'n02105641', 'n02096051', 'n02110341', 'n02086910', 'n02113712', 'n02113186', 'n02091467',
           'n02106550', 'n02091831', 'n02104365', 'n02086079']


def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def get_threshold_mask(h, w, r):
    center = (int(h / 2), int(w / 2))
    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            #mask[i, j] = np.exp(((-distance((i,j),(ch, cw))**2)/(2*(50**2))))
            # if np.sqrt((i - ch) ** 2 + (j - cw)**2) < r:
            if distance((i, j), center) < r:
                mask[i, j] = 1.0
    return mask


def get_gaussian_mask(h, w, r):
    center = (int(h / 2), int(w / 2))
    mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            mask[i, j] = np.exp(-(distance((i, j), center)**2)/(2*(r**2)))
    return mask


def fourier_transofrmation(img, r, mode='low'):
    h, w, c = img.shape
    mask = get_gaussian_mask (h, w, r)
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
    for wnid in SIMILAR:
        paths = glob(f'../data/imagenet/train/{wnid}/*')
        os.makedirs(f'../data/imagenet_freq_similar/G_12_lfc/train/{wnid}', exist_ok=True)
        os.makedirs(f'../data/imagenet_freq_similar/G_12_hfc/train/{wnid}', exist_ok=True)

        for path in paths:
            img_name = path.split('/')[-1]
            origin = cv2.imread(path)
            # spec_4, lfc_4, hfc_4 = fourier_transofrmation(img, r=4, mode='both')
            # _, lfc_8, hfc_8 = fourier_transofrmation(img, r=8, mode='both')
            # _, lfc_12, hfc_12 = fourier_transofrmation(img, r=12, mode='both')
            # _, lfc_16, hfc_16 = fourier_transofrmation(img, r=16, mode='both')

            origin_spec, origin_lfc, origin_hfc = fourier_transofrmation(origin, r=12, mode='both')
            cv2.imwrite(f'../data/imagenet_freq_similar/G_12_lfc/train/{wnid}/{img_name}', origin_lfc)
            cv2.imwrite(f'../data/imagenet_freq_similar/G_12_hfc/train/{wnid}/{img_name}', origin_hfc)
            # img_spec, img_lfc, img_hfc = fourier_transofrmation(img, r=10, mode='both')

            # cv2.imshow("origin", origin)
            # cv2.imshow("img", img)
            #
            # cv2.imshow("img_spec", img_spec)
            # cv2.imshow('img_lfc', img_lfc)
            # cv2.imshow('img_hfc', img_hfc)
            #
            # cv2.imshow("origin_spec", origin_spec)
            # cv2.imshow('origin_lfc', origin_lfc)
            # cv2.imshow('origin_hfc', origin_hfc)
            #
            # resized_spec = cv2.resize(origin_spec, (224, 224))
            # resized_lfc = cv2.resize(origin_lfc, (224, 224))
            # resized_hfc = cv2.resize(origin_hfc, (224, 224))
            # cv2.imshow("resized_spec", resized_spec)
            # cv2.imshow('resized_lfc', resized_lfc)
            # cv2.imshow('resized_hfc', resized_hfc)
            #
            # print(np.min(img_spec), np.max(img_spec))
            # print(np.min(resized_spec), np.max(resized_spec))
            # print(np.min(img_lfc), np.max(img_lfc))
            # print(np.min(resized_lfc), np.max(resized_lfc))
            # print(np.min(img_hfc), np.max(img_hfc))
            # print(np.min(resized_hfc), np.max(resized_hfc))
            #
            # print(np.mean(resized_spec - img_spec))
            # print(np.mean(resized_lfc - img_lfc))
            # print(np.mean(resized_hfc - img_hfc))

            # cv2.imshow("spec", np.uint8(spec_4))

            # cv2.imshow("lfc_4", lfc_4)
            # cv2.imshow("hfc_4", hfc_4)
            # cv2.imshow("lfc_8", lfc_8)
            # cv2.imshow("hfc_8", hfc_8)
            # cv2.imshow("lfc_12", lfc_12)
            # cv2.imshow("hfc_12", hfc_12)
            # cv2.imshow("lfc_16", lfc_16)
            # cv2.imshow("hfc_16", hfc_16)

        print(wnid)
