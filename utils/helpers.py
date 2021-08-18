import os

import numpy as np
import torch

from torchvision.utils import make_grid

import imageio

import pylab
from skimage.measure import compare_ssim

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def create_dir(dir):
    """
    Creates a directory
    """
    try:
        os.makedirs(dir)
    except:
        print('folder '+dir+' exists')


def save_image_numpy(im, path):
    """
    Saves an image im np format
    """
    im_bounded = im*255.
    im_bounded[im_bounded > 255.] = 255.
    im_bounded[im_bounded < 0.] = 0.
    imageio.imwrite(path, np.uint8(im_bounded))


def save_image(Img,path,nrows=10):
    """
    Saves an image im torch format
    """
    if isinstance(Img, torch.Tensor):
        with torch.no_grad():
            Img = make_grid(Img, nrow=nrows, normalize=True, scale_each=True)
            try:
                Img = Img.cpu().numpy()
            except:
                Img = Img.detach().cpu().numpy()
    Img = np.moveaxis(Img, 0, -1)
    Img[Img > 1.] = 1.
    Img[Img < 0.] = 0.
    imageio.imwrite(path, np.uint8(Img*255.))


def save_several_images(ims, paths, nrows=10, out_format='.png'):
    """
    Saves an image im torch format
    """
    for i, im in enumerate(ims):
        create_dir(os.path.dirname(paths[i]))
        save_image(im, paths[i]+out_format, nrows=nrows)


def snr(x, y):
    """
    snr - signal to noise ratio

       v = snr(x,y);

     v = 20*log10( norm(x(:)) / norm(x(:)-y(:)) )

       x is the original clean signal (reference).
       y is the denoised signal.

    Copyright (c) 2014 Gabriel Peyre
    """
    return 20 * np.log10(pylab.norm(x.flatten()) / (pylab.norm(x.flatten() - y.flatten())+1e-6))


def snr_torch(x, y):
    """
    snr - signal to noise ratio

       v = snr(x,y);

     v = 20*log10( norm(x(:)) / norm(x(:)-y(:)) )

       x is the original clean signal (reference).
       y is the denoised signal.
    """
    s = 0
    for _ in range(x.shape[0]):
        x_np = x[_, ...].detach().cpu().numpy()
        y_np = y[_, ...].detach().cpu().numpy()
        s += snr(x_np, y_np)
    return s/float(x.shape[0])

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += compare_psnr(Iclean[i, ...], Img[i, ...], data_range=data_range)
    return psnr/Img.shape[0]


def compute_metrics(true_image, denoised_image):
    if isinstance(true_image, torch.Tensor):
        avg_snr = snr_torch(true_image, denoised_image)
        true_image_np, denoised_image_np = true_image[0].cpu().numpy(), denoised_image[0].cpu().numpy()
        true_image_np, denoised_image_np = np.moveaxis(true_image_np, 0, -1), np.moveaxis(denoised_image_np, 0, -1)
        avg_ssim = compare_ssim(true_image_np, denoised_image_np, gaussian_weights=True, data_range=1, multichannel=True)
        mse = torch.mean((denoised_image - true_image) ** 2) # Taken from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
        avg_psnr = 10 * torch.log10(1 / mse)
    else:
        avg_snr = snr(true_image, denoised_image)
        avg_ssim = compare_ssim(np.moveaxis(true_image, 0, -1), np.moveaxis(denoised_image, 0, -1), gaussian_weights=True, data_range=1, multichannel=True)
        mse = np.mean((denoised_image - true_image) ** 2)
        avg_psnr = 10*np.log10(1/mse)
    return avg_snr, avg_psnr, avg_ssim
