import numpy as np
import torch

from optim import FB, get_operators
from utils.helpers import create_dir

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def prox_test(y_np, forward_op, backward_op, x_true=None, max_iter=100, gamma=1., folder_save='images/', noise_level=0.05, save_pictures=False, crit_conv=None, denoiser=None, save_info=True):
    """
    Takes as input a batch of torch images and applies the FB algorithm ; returns the solution.
    """

    if save_info:
        create_dir(folder_save)

    sol, best, x_diff_list, loss_list, x_norm_list, loss_rel, lip_list, snr_list = FB(y_np, forward_op, backward_op, denoiser, x_true=x_true, gamma=gamma, folder_save=folder_save, max_iter=max_iter, save_pictures=save_pictures, crit_conv=crit_conv)

    x_norms = x_diff_list[:, None]
    loss_ = loss_list[:, None]
    loss_rel_ = loss_rel[:, None]
    x_norms_nb = x_norm_list[:, None]
    snrs = snr_list[:, None]

    x_n_tot = x_norms
    x_nobord_tot = x_norms_nb
    loss_tot = loss_
    loss_rel_tot = loss_rel_
    snr_tot = snrs

    x_n_av = np.mean(x_n_tot, axis=1)
    x_n_std = np.std(x_n_tot, axis=1)

    if save_info:
        np.savetxt(folder_save+'xn_tot_'+str(noise_level)+'.txt', x_n_tot)
        np.savetxt(folder_save+'xn_diff_'+str(noise_level)+'.txt', x_n_av) 
        np.savetxt(folder_save+'xn_std_'+str(noise_level)+'.txt', x_n_std)
        np.savetxt(folder_save+'xn_norm_'+str(noise_level)+'.txt', x_nobord_tot)
        np.savetxt(folder_save+'loss_tot_'+str(noise_level)+'.txt', loss_tot)
        np.savetxt(folder_save+'loss_rel_tot_'+str(noise_level)+'.txt', loss_rel_tot)
        np.savetxt(folder_save+'snr_tot_'+str(noise_level)+'.txt', snr_tot)

    return sol, y_np, best


def prox_test_torch(im_torch, type_pb='circular_deconvolution', max_iter=100, gamma=1., folder_save='images/', noise_level=0.05, net_name='DnCNN_nobn', save_pictures=False, pth_kernel=None, crit_conv=None, denoiser=None, cross_cor=False, save_info=True):
    """
    takes as input a batch of torch images and applies the FB algorithm ; returns the solution
    """

    torch_sol = torch.zeros_like(im_torch)
    torch_best = torch.zeros_like(im_torch)
    torch_blu = torch.zeros_like(im_torch)
    create_dir(folder_save)

    shape_im = im_torch[0, ...].cpu().numpy().shape

    forward_op, backward_op, noise = get_operators(type_op=type_pb, sigma=noise_level, pth_kernel=pth_kernel, shape=shape_im, cross_cor=cross_cor)

    for _ in range(im_torch.shape[0]):

        im_np = im_torch[_, ...].cpu().numpy()

        x_blurred = forward_op(im_np)
        y = x_blurred+noise

        sol, best, x_diff_list, loss_list, x_norm_list, loss_rel, lip_list, snr_list = FB(y, forward_op, backward_op, denoiser, x_true=im_np, gamma=gamma, folder_save=folder_save, max_iter=max_iter, save_pictures=save_pictures, crit_conv=crit_conv)
        
        sol_torch_cur = torch.from_numpy(sol)
        sol_torch_cur.unsqueeze_(0)

        blu_torch_cur = torch.from_numpy(y)
        blu_torch_cur.unsqueeze_(0)

        best_torch_cur = torch.from_numpy(best)
        best_torch_cur.unsqueeze_(0)

        torch_sol[_] = sol_torch_cur
        torch_blu[_] = blu_torch_cur
        torch_best[_] = best_torch_cur

        x_norms = x_diff_list[:, None]
        loss_ = loss_list[:, None]
        loss_rel_ = loss_rel[:, None]
        x_norms_nb = x_norm_list[:, None]
        lip_cts = lip_list[:, None]
        snrs = snr_list[:, None]

        if _ == 0:
            x_n_tot = x_norms
            x_nobord_tot = x_norms_nb
            loss_tot = loss_
            loss_rel_tot = loss_rel_
            lip_tot = lip_cts
            snr_tot = snrs

        else:
            x_n_tot = np.hstack((x_n_tot, x_norms))
            x_nobord_tot = np.hstack((x_nobord_tot, x_norms_nb))
            loss_tot = np.hstack((loss_tot, loss_))
            loss_rel_tot = np.hstack((loss_rel_tot, loss_rel_))
            lip_tot = np.hstack((lip_tot, lip_cts))
            snr_tot = np.hstack((snr_tot, snrs))

        x_n_av = np.mean(x_n_tot, axis=1)
        x_n_std = np.std(x_n_tot, axis=1)

        if save_info:
            np.savetxt(folder_save+'xn_tot_'+str(noise_level)+'.txt', x_n_tot)
            np.savetxt(folder_save+'xn_diff_'+str(noise_level)+'.txt', x_n_av) 
            np.savetxt(folder_save+'xn_std_'+str(noise_level)+'.txt', x_n_std)
            np.savetxt(folder_save+'xn_norm_'+str(noise_level)+'.txt', x_nobord_tot)
            np.savetxt(folder_save+'loss_tot_'+str(noise_level)+'.txt', loss_tot)
            np.savetxt(folder_save+'loss_rel_tot_'+str(noise_level)+'.txt', loss_rel_tot)
            np.savetxt(folder_save+'snr_tot_'+str(noise_level)+'.txt', snr_tot)

    return torch_sol, torch_blu, torch_best
