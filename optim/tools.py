import numpy as np

import torch
import torch.nn as nn

from models import simple_CNN
from utils import save_image_numpy, snr

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Denoiser:
    """
    This class shall encompass all possible denoisers (networks, prox operators, BM3D...)
    It contains a "denoise" method that applies the effective denoising function.
    """
    def __init__(self, model=None, nature='network', arch='FFCNN', channels=3, path=None, function=None, parameters=None, cuda=True, sigma=0.01, root_path='.', problem_type=None):
        """ 
        Intialise the denoiser
        """
        self.nature = nature
        self.arch = arch
        self.sigma = sigma
        self.problem_type = problem_type
        
        if self.nature == 'network':
            if model is None:
                self.network = load_net(path, net_type=arch, channels=channels, cuda=cuda, root_folder=root_path, n_lev=self.sigma)       # Path of the saved model in the case the denoiser we use is a network
            else:
                self.network = model

        if self.nature == 'function':
            self.parameters = parameters
            self.function = function

        self.cost = 0
            
    def denoise(self, x):
        """
        Returns a denoised version of the image.
        Assumes that x has a torch image format (C,W,H) withouth batch dimension
        """

        if self.nature == 'network':
            return apply_model(x, model=self.network)
        
        if self.nature == 'function':
            params = self.parameters

            if len(x.shape) == 3:  # If the function was meant to work on a regular numpy formatting (i.e. (W,H,C)) then we need to flip axes
                x = np.moveaxis(x, 0, -1)

            if params is not None:
                out, cost = self.function(x, *params)
            else:
                out, cost = self.function(x)
            self.cost = cost

            if len(x.shape) == 3:
                out = np.moveaxis(out, -1, 0)

            return out


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.module.load_state_dict(checkpoint.module.state_dict())
    return model


def load_net(pth=None, net_type='DnCNN_nobn', channels=3, n_lev=0.01, cuda=True, root_folder='.'):

    if 'DnCNN_nobn' in net_type:
        avg, bn, depth = False, False, 20
        net = simple_CNN(n_ch_in=channels, n_ch_out=channels, n_ch=64, nl_type='relu', depth=depth, bn=bn)
        pth = root_folder+'checkpoints/pretrained/'+net_type+'_nch_'+str(channels)+'_nlev_'+str(n_lev)+'.pth'
        
    if cuda:
        cuda_infotxt = "cuda driver found - moving to GPU.\n"
        print(cuda_infotxt)
        net.cuda()

    if cuda:
        model = nn.DataParallel(net).cuda()
    else:
        model = nn.DataParallel(net)
            
    if pth is not None:
        loaded_txt = "Loading " + pth + "...\n"
        print(loaded_txt)
        model = load_checkpoint(model, pth)
    else:
        raise NameError('Could not load '+str(net_type))
    
    return model.eval() 
            

def apply_model(x_cur, model=None):

    imgn = torch.from_numpy(x_cur)
    init_shape = imgn.shape
    if len(init_shape) == 2:  
        imgn.unsqueeze_(0)
        imgn.unsqueeze_(0)
    elif len(init_shape) == 3:
        imgn.unsqueeze_(0)
    imgn = imgn.type(Tensor)                          

    with torch.no_grad():
        imgn.clamp_(0, 1)  # Might be necessary for some networks
        out_net = model(imgn)
        out_net.clamp_(0, 1)

    img = out_net[0, ...].cpu().detach().numpy()
    if len(init_shape) == 2:
        x = img[0]     
    elif len(init_shape) == 3:
        x = img                 

    return x 


def get_denoiser(denoiser_type, nature='network', n_ch=1, cuda=True, root_path='/lustre/home/sc004/mterris/python_experiments/pnp_experiments_inria/', sigma=None, problem_type=None):

    denoiser = Denoiser(nature=nature, arch=denoiser_type, channels=n_ch, cuda=cuda, root_path=root_path, sigma=sigma, problem_type=problem_type)

    return denoiser


def FB(y, forward_op, backward_op, denoiser, gamma=1., nu=1., max_iter=100, save_pictures=True, folder_save='images/', crit_conv=None, x_true=None):
    """
    PnP FB algorithm
    """

    # Logs
    x_diff_list = np.zeros(max_iter)
    x_norm_list = np.zeros(max_iter)
    loss_list = np.zeros(max_iter)
    loss_rel = np.zeros(max_iter)
    lip_list = np.zeros(max_iter)
    snr_list = np.zeros(max_iter)
    
    # Initialisation
    x = np.copy(y)
    best = np.copy(x)
    y_ref = np.copy(y)
    temp = 1.
    best_snr = 0.

    for it in range(max_iter):
        
        x_prev = np.copy(x)
        temp_prev = np.copy(temp)

        temp = forward_op(x)-y_ref 
        grad = backward_op(temp)
        x_cur = x-gamma*grad 
        den = denoiser.denoise(x_cur)
        x = (1-nu)*x+nu*den
            
        if save_pictures:
            save_name = folder_save+'x_'+str(it)+'_test.png'
            save_image_numpy(x, save_name)

        x_diff = np.linalg.norm(x.flatten()-x_prev.flatten())
        x_norm = np.linalg.norm(x_prev.flatten())
        l_n = np.linalg.norm(temp)/np.linalg.norm(y_ref)
        l_n_rel = (np.linalg.norm(temp)-np.linalg.norm(temp_prev))/np.linalg.norm(temp_prev)

        if x_true is not None:
            s_cur = snr(x_true, x)
        else:
            s_cur = 0.

        if s_cur > best_snr:
            best = np.copy(x_cur)
            best_snr = s_cur

        x_diff_list[it] = x_diff
        x_norm_list[it] = x_norm
        loss_list[it] = l_n
        loss_rel[it] = l_n_rel
        lip_list[it] = 0
        snr_list[it] = s_cur

        if crit_conv is not None and x_diff/x_norm < crit_conv:
            break

    return x, best, x_diff_list, loss_list, x_norm_list, loss_rel, lip_list, snr_list
