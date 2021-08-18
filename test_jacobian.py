import argparse
import glob
import os
import json

import numpy as np
from PIL import Image

import torch

from utils.helpers import create_dir, compute_metrics
from models import get_model
from jacobian import JacobianReg_l2


parser = argparse.ArgumentParser(description="testing pnp")
parser.add_argument("--n_ch", type=int, default=3, help="number of channels")
parser.add_argument("--noise_level", type=float, default=0.01, help='noise level')
parser.add_argument("--noise_level_den", type=float, default=0.009, help='noise level')
parser.add_argument("--gamma", type=float, default=1.0, help='noise level')
parser.add_argument("--kernel", type=str, default='blur_1', help='kernel of the degradation measurement operator')
parser.add_argument("--architecture", type=str, default='DnCNN_nobn', help="type of network to study. Should be DnCNN_nobn or DnCNN for now")
parser.add_argument("--pth_config_file", type=str, default='configfiles/setup.json')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.backends.cudnn.benchmark = True


def eval_jacobian(architecture='DnCNN_nobn', n_ch=3, n_lev=0.01, noise_level_den=0.01, max_iter=5, pth_config_file='configfiles/setup.json'):

    np.random.seed(0)

    dir_text = 'pnp_benchmark/jacobian_eval/'+architecture+'_nch_'+str(n_ch)
    img_folder = dir_text
    
    create_dir(dir_text)
    create_dir(img_folder)

    with open(pth_config_file, 'r') as f:
        config = json.load(f)

    path_dataset_pnp = config['path_dataset_pnp']
    pattern_red = config['pattern_red']

    pth_images = sorted(glob.glob(os.path.join(path_dataset_pnp, pattern_red)))

    n_f = 'val_info_'+str(n_lev)+'.txt'

    text_file = open(dir_text+'/'+n_f, "w+")
    text_file.write('''Testing jacobian for {}:\nCUDA: {}\n \n sigma, SNR_in, SNR, PSNR_in, PSNR, SSIM_in, SSIM, mean_FNE,
     max_FNE, mean_lip, max_lip \n'''.format(architecture, cuda))
    text_file.close()

    reg_fun_val = JacobianReg_l2(eval_mode=True, max_iter=max_iter)

    model, _, _, _ = get_model(architecture, n_ch=n_ch)
    checkpoint = torch.load('checkpoints/pretrained/DnCNN_nobn_nch_'+str(n_ch)+'_nlev_'+str(noise_level_den)+'.pth', map_location=lambda storage, loc: storage)
    model.module.load_state_dict(checkpoint.module.state_dict())
    model.eval()

    max_lip, max_fne, avg_lip, avg_fne = 0, 0, 0, 0
    snr_in_avg, psnr_in_avg, ssim_in_avg, snr_avg, psnr_avg, ssim_avg = 0, 0, 0, 0, 0, 0

    for i, image_path in enumerate(pth_images, 0):

        image_true = Image.open(image_path)
        image_true = np.asarray(image_true, dtype="float32")/255.
        image_true = np.moveaxis(image_true, -1, 0)
        if n_ch == 1:
            image_true = (image_true[0:1, ...]+image_true[1:2, ...]+image_true[2:3, ...])/3.
        image_true = image_true[:, :40, :40]

        torch.cuda.empty_cache()  # Commented out for realsn

        # Evaluating the FNE
        torch_im = torch.from_numpy(image_true).unsqueeze(0).type(Tensor)  # Moving to torch
        data_true = torch_im.requires_grad_()  # Activate requires_grad for backprop
        noise = n_lev*torch.randn_like(data_true)
        data_noisy = data_true + noise

        out = model(data_noisy)
            
        out_q = 2.*out-data_noisy
        jac_norm = reg_fun_val(data_noisy, out_q)

        if jac_norm.item() > max_fne:
            max_fne = jac_norm.item()
        avg_fne = avg_fne+jac_norm.item()

        lip_net = reg_fun_val(data_noisy, out)

        if lip_net > max_lip:
            max_lip = lip_net.item()
        avg_lip = avg_lip+lip_net.item()

        data_true.detach_()
        data_noisy.detach_()
        out.detach_()

        snr_in, psnr_in, ssim_in = compute_metrics(data_true, data_noisy)
        snr, psnr, ssim = compute_metrics(data_true, out)

        snr_in_avg, psnr_in_avg, ssim_in_avg = snr_in_avg + snr_in, psnr_in_avg + psnr_in, ssim_in_avg + ssim_in
        snr_avg, psnr_avg, ssim_avg = snr_avg + snr, psnr_avg + psnr, ssim_avg + ssim

    snr_in_avg, psnr_in_avg, ssim_in_avg = snr_in_avg/(i+1), psnr_in_avg/(i+1), ssim_in_avg/(i+1)
    snr_avg, psnr_avg, ssim_avg = snr_avg/(i+1), psnr_avg/(i+1), ssim_avg/(i+1)
    avg_fne, avg_lip = avg_fne/(i+1), avg_lip/(i+1)

    text_file = open(dir_text+'/'+n_f, 'w+')  # Creating the text file with training infos Other infos would be relevant such as stride, number of iterations in proj etc
    text_file.write('''noise level, snr_in, snr_out, psnr_in, psnr_out, ssim_in, ssim_out, avg_fne, max_fne, avg_lip, max_lip
{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'''.format(n_lev, snr_in_avg, snr_avg, psnr_in_avg, psnr_avg, ssim_in_avg, ssim_avg, avg_fne, max_fne, avg_lip, max_lip))
    text_file.close() 


if __name__ == "__main__":
    
    eval_jacobian(n_ch=opt.n_ch, n_lev=opt.noise_level, noise_level_den=opt.noise_level_den, max_iter=5, architecture=opt.architecture, pth_config_file=opt.pth_config_file)
