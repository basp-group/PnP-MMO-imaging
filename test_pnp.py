import argparse
import glob
import os
import json

import numpy as np
from PIL import Image

import torch
from utils.helpers import create_dir, save_several_images, compute_metrics

from prox_solver import prox_test
from optim import Denoiser, get_operators

parser = argparse.ArgumentParser(description="testing pnp")
parser.add_argument("--n_ch", type=int, default=3, help="number of channels")
parser.add_argument("--noise_level", type=float, default=0.01, help='noise level')
parser.add_argument("--noise_level_den", type=float, default=0.007, help='noise level; set 0.007 for color, 0.009 for grayscale.')
parser.add_argument("--gamma", type=float, default=1.99, help='noise level')
parser.add_argument("--kernel", type=str, default='blur_1', help='kernel of the degradation measurement operator')
parser.add_argument("--architecture", type=str, default='DnCNN_nobn', help="type of network to study. Should be DnCNN_nobn or DnCNN for now")
parser.add_argument("--pth_config_file", type=str, default='configfiles/setup.json')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.backends.cudnn.benchmark = True


def eval_pnp(architecture='DnCNN_nobn', n_ch=3, n_lev=0.01, gamma=1.0, noise_level_den=0.01, max_iter=100, type_pb='circular_deconvolution', pth_kernel=None, crit_conv=1e-5, pth_config_file='configfiles/setup.json'):

    np.random.seed(0)

    type_pb = type_pb + '_' + os.path.splitext(os.path.basename(pth_kernel))[0]

    dir_text = 'pnp_benchmark/FB_nch_'+str(n_ch)+'/'+architecture
    img_folder = dir_text
    
    create_dir(dir_text)
    create_dir(img_folder)

    with open(pth_config_file, 'r') as f:
        config = json.load(f)

    path_dataset_pnp = config['path_dataset_pnp']
    pattern_red = config['pattern_red']
    root_folder = config['root_folder']

    pth_images = sorted(glob.glob(os.path.join(path_dataset_pnp, pattern_red)))

    n_f = 'val_info_'+str(n_lev)+'.txt'

    text_file = open(dir_text+'/'+n_f, "w+")
    text_file.write('''Testing PnP for {}:\nCUDA: {}\n'''.format(architecture, cuda))
    text_file.close()

    snr_avg, psnr_avg, ssim_avg = 0, 0, 0 

    denoiser = Denoiser(arch=architecture, nature='network', channels=n_ch, cuda=cuda, sigma=noise_level_den, root_path=root_folder)

    for i, image_path in enumerate(pth_images, 0):

        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]

        image_true = Image.open(image_path)
        image_true = np.asarray(image_true, dtype="float32")/255.
        image_true = np.moveaxis(image_true, -1, 0)
        if not cuda:
            image_true = image_true[:, :64, :64]  # The algorithm can be slow without GPUs.
        if n_ch == 1:
            image_true = (image_true[0:1, ...]+image_true[1:2, ...]+image_true[2:3, ...])/3.

        img_folder_cur = img_folder+'/'+image_name+'/'

        forward_op, backward_op, _ = get_operators(type_op=type_pb, sigma=None, pth_kernel=pth_kernel, shape=image_true.shape, cross_cor=True)

        x_blurred = forward_op(image_true)
        noise = np.random.randn(*image_true.shape)
        y = x_blurred+n_lev*noise

        sol_pnp, blu_pnp, best_pnp = prox_test(y, forward_op, backward_op, x_true=None, denoiser=denoiser, gamma=gamma, max_iter=max_iter, folder_save=img_folder_cur, noise_level=n_lev, crit_conv=crit_conv)  # Takes a batch as input, returns a batch

        cur_snr, cur_psnr, cur_ssim = compute_metrics(image_true[..., 6:-6, 6:-6], sol_pnp[..., 6:-6, 6:-6])
        best_snr, best_psnr, best_ssim = compute_metrics(image_true[..., 6:-6, 6:-6], best_pnp[..., 6:-6, 6:-6])

        pictures = [sol_pnp, image_true, blu_pnp, best_pnp]
        pic_paths = [img_folder_cur+'sol_'+str(n_lev), img_folder_cur+'target_'+str(n_lev), img_folder_cur+'blurred_'+str(n_lev),  img_folder_cur+'best_'+str(n_lev)]
        save_several_images(pictures, pic_paths, nrows=int(np.sqrt(sol_pnp.shape[0])), out_format='.png')

        text_file = open(dir_text+'/'+n_f, "a")
        text_file.write('{} snr_avg: {:.3f} psnr_avg: {:.3f} ssim_avg: {:.3f} PSNR_best: {:.3f} SSIM_best: {:.3f}\n'.format(image_name, cur_snr, cur_psnr, cur_ssim, best_psnr, best_ssim))
        text_file.close()

        snr_avg = snr_avg + cur_snr
        ssim_avg = ssim_avg + cur_ssim
        psnr_avg = psnr_avg + cur_psnr

    snr_avg, psnr_avg, ssim_avg = snr_avg/(i+1), psnr_avg/(i+1), ssim_avg/(i+1)

    text_file = open(dir_text+'/'+'avg.txt', 'w+')  # Creating the text file with training infos Other infos would be relevant such as stride, number of iterations in proj etc
    text_file.write('''{:.2f} {:.2f} {:.2f} {:.4f}\n'''.format(n_lev, snr_avg, psnr_avg, ssim_avg))
    text_file.close() 


if __name__ == "__main__":
    
    blurs = ['blur_models/'+opt.kernel+'.mat']
    if opt.kernel == 'gaussian_1_6':
        noises = [0.008]
    else:
        noises = [0.01]

    for pth_kernel, n_lev in zip(blurs, noises):
        eval_pnp(gamma=opt.gamma, n_ch=opt.n_ch, n_lev=n_lev, max_iter=300, noise_level_den=opt.noise_level_den, architecture=opt.architecture, pth_kernel=pth_kernel, crit_conv=None, pth_config_file=opt.pth_config_file)
