import argparse
import os
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils.im_class import get_dataset
from utils.helpers import create_dir, save_several_images, snr_torch
from models import get_model

from prox_solver import prox_test_torch as prox_test
from jacobian import JacobianReg_l2
from optim import Denoiser

parser = argparse.ArgumentParser(description="training")
parser.add_argument("-c", "--n_ch", type=int, default=3, help="number of channels")
parser.add_argument("-b", "--batchSize", type=int, default=50, help="Training batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lambdajr", type=float, default=1e-5, help="Initial reg parameter")
parser.add_argument("--outf", type=str, default="checkpoints", help='path of log files')
parser.add_argument("--noise_level", type=float, default=0.01, help='noise level')
parser.add_argument("--epsilon", type=float, default=0., help='safety bound for lip (should be positive)')
parser.add_argument("--architecture", type=str, default='DnCNN_nobn', help="type of network to study.")
parser.add_argument("--pth_config_file", type=str, default='configfiles/setup.json')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.backends.cudnn.benchmark = True 


def compute_reg(out, data_true, model, reg_fun, epsilon):
    """
    Computes the regularization reg_fun applied to the correct point
    """
    out_detached = out.detach().type(Tensor)
    true_detached = data_true.detach().type(Tensor)

    tau = torch.rand(true_detached.shape[0], 1, 1, 1).type(Tensor)
    out_detached = tau*out_detached+(1-tau)*true_detached
    out_detached.requires_grad_()

    out_reg = model(out_detached)
        
    out_net_reg = 2.*out_reg-out_detached
    reg_loss = reg_fun(out_detached, out_net_reg)
    reg_loss_max = torch.max(reg_loss, torch.ones_like(reg_loss)-epsilon)
    return reg_loss_max.max() 


def train_net(architecture='DnCNN_nobn', n_ch=1, epsilon=0., lambda_jr=1e-5, noise_lev=0.01, pth_config_file='configfiles/setup.json'):

    torch.manual_seed(0)  # Reproducibility

    str_id = 'epsilon_'+str(epsilon)+'_nch_'+str(n_ch)+'_ljr_'+str(lambda_jr)+'_'+str(noise_lev)

    # Creating the necessary folders
    dir_text = 'infos_training/'+architecture+'/'  # This folder will contain our training logs
    img_folder = 'images_test/'+architecture+'/'+str_id+'/'
    img_folder_train = img_folder+'/training/'
    img_folder_pnp = img_folder+'/pnp/'
    dir_checkpoint = opt.outf+'/'+architecture+'/'
    
    for f in [dir_text, img_folder, img_folder_train, img_folder_pnp, dir_checkpoint]:
        create_dir(f)

    # Creating the log files
    n_f, n_f_sum = 'training_info_'+str_id+'.txt', 'training_info_'+str_id+'_summary.txt'

    for f_text in [n_f, n_f_sum]:
        text_file = open(dir_text+'/'+f_text, "w+")
        text_file.write('Launching...\n')
        text_file.close()
    
    # Loading our dataset
    with open(pth_config_file, 'r') as f:
        config = json.load(f)

    path_dataset = config['path_dataset']
    path_dataset_pnp = config['path_dataset_pnp']
    pattern_red = config['pattern_red']

    loader_train, loader_val, loader_pnp = get_dataset(path_dataset=path_dataset, path_red_dataset=path_dataset_pnp, patchSize=45, red=True, red_size=10, channels=n_ch, bs=opt.batchSize, pattern_red=pattern_red)

    # Loading our model
    model, net_name, clip_val, lr = get_model(architecture, n_ch=n_ch)
    net_name = net_name+str_id
        
    text_file = open(dir_text+'/'+n_f, "a")  # Creating the log
    text_file.write('''                  
Starting training {}:
Epochs: {}
Batch size: {}
Learning rate: {}
Training size: {}
Validation size: {}
Pnp val size:{}
Checkpoints: {}
CUDA: {}
'''.format(architecture, opt.epochs, opt.batchSize, lr, len(loader_train),
    len(loader_val), len(loader_pnp), opt.outf, cuda))
    text_file.close()

    # Defining the loss and Jacobian regularisation losses (one for training and one memory efficient for validation/test)
    criterion = nn.MSELoss(reduction='sum')
    reg_fun, reg_fun_val = JacobianReg_l2(), JacobianReg_l2(eval_mode=True)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    loss_val_best = 1e12

    for epoch in range(opt.epochs):

        loss_tot, avg_snr_train, avg_snr_noise, max_jac_train = 0, 0, 0, 0.

        torch.cuda.empty_cache()

        ########################
        # TRAINING
        ########################

        model.train()

        for i, data in enumerate(loader_train, 0):

            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            data_true = data['image_true'][:, :n_ch, ...].type(Tensor)
            noise = noise_lev*torch.randn_like(data_true)

            data_noisy = data_true+noise

            out = model(data_noisy)
            out_net = 2.*out-data_noisy

            size_tot = data_true.size()[0]*2*data_true.size()[-1]**2
            loss = criterion(out, data_true)/size_tot

            reg_loss_max = compute_reg(out, data_true, model, reg_fun, epsilon=epsilon)
            reg_loss_val = lambda_jr*reg_loss_max

            if reg_loss_max.max() > max_jac_train:
                max_jac_train = reg_loss_max.max().item()

            loss_reg = loss+reg_loss_val

            # Original
            loss_reg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            snr_in = snr_torch(data_true, data_noisy)
            snr_train = snr_torch(data_true, out)

            if np.isfinite(snr_train) and np.isfinite(snr_in):
                avg_snr_train += snr_train
                avg_snr_noise += snr_in

            text_file = open(dir_text+'/'+n_f, "a")
            text_file.write('[epoch {}][{}/{}] loss: {:.3e} reg: {:.3e} den: {:.3e} SNRin: {:.2f} SNRout: {:.2f} \n'.format(epoch+1, i+1, len(loader_train), loss_reg.item(), reg_loss_max.item(), loss.item(), snr_in, snr_train))
            text_file.close()

            if i % 150 == 0:  # Save some training images
                pictures = [data_noisy, data_true, out, out_net]
                pic_paths = [img_folder_train+'/Img_in', img_folder_train+'/Img_true', img_folder_train+'/Img_out', img_folder_train+'/Img_out_net']
                save_several_images(pictures, pic_paths, nrows=10, out_format='.png')

            loss_tot = loss_tot + loss_reg.item()

        loss_tot_avg = loss_tot/(i+1)
        avg_snr_train = avg_snr_train/(i+1)
        avg_snr_noise = avg_snr_noise/(i+1)

        ########################
        # VALIDATION
        ########################

        model.eval()
        loss_val_tot, avg_snr_val, max_jac_val, avg_snr_val_noise = 0., 0., 0., 0.

        for i_val, data in enumerate(loader_val, 0):
                
            torch.cuda.empty_cache()  # Commented out for realsn

            data_true = data['image_true'][:, :n_ch, ...].type(Tensor).requires_grad_()
            noise = noise_lev*torch.randn_like(data_true)
            data_noisy = data_true+noise

            out = model(data_noisy)
            out_net = 2.*out-data_noisy

            with torch.no_grad():
                size_tot = data_true.size()[0]*2*data_true.size()[-1]**2
                loss = criterion(out, data_true)/size_tot

            reg_loss_max = compute_reg(out, data_true, model, reg_fun_val, epsilon=epsilon)
            reg_loss_max = reg_loss_max.mean()

            if reg_loss_max.max() > max_jac_val:
                max_jac_val = reg_loss_max.max().item()

            loss_reg = loss+lambda_jr*reg_loss_max

            snr_in = snr_torch(data_true, data_noisy)
            snr_val = snr_torch(data_true, out)

            avg_snr_val += snr_val
            avg_snr_val_noise += snr_in

            if i_val == 0:  # Save some validation images
                pictures = [data_noisy, data_true, out, out_net]
                pic_paths = [img_folder_train+'/Img_in_val', img_folder_train+'/Img_true_val', img_folder_train+'/Img_out_val', img_folder_train+'/Img_out_net_val']
                save_several_images(pictures, pic_paths, nrows=10, out_format='.png')
            
            loss_val_tot = loss_val_tot + loss_reg.item()

        loss_val_tot_avg = loss_val_tot/(i_val+1)
        avg_snr_val = avg_snr_val/(i_val+1)
        avg_snr_val_noise = avg_snr_val_noise/(i_val+1)

        text_file = open(dir_text+'/'+n_f_sum, "a")
        text_file.write('[epoch {}] Average training loss: {:.3e} Average training input SNR: {:.2f} Average training SNR: {:.2f} Max jacobian training: {:.3e} Average validation loss: {:.3e} Average validation input SNR: {:.2f} Average validation SNR: {:.2f} Max jacobian validation: {:.3e} \n'.format(epoch+1, loss_tot_avg, avg_snr_noise, avg_snr_train, max_jac_train, loss_val_tot_avg, avg_snr_val_noise, avg_snr_val, max_jac_val))
        text_file.close()

        scheduler.step()

        # Save the model
        torch.save(model, os.path.join(dir_checkpoint, net_name+'.pth'))
        if epoch % 50 == 0:
            torch.save(model, os.path.join(dir_checkpoint, net_name+'_e'+str(epoch)+'.pth'))
        if loss_val_tot_avg < loss_val_best:
            torch.save(model, os.path.join(dir_checkpoint, net_name+'_best.pth'))
            loss_val_best = loss_val_tot_avg

        ########################
        # VALIDATE PNP
        ########################
        
        if epoch % 10 == 0:
            denoiser = Denoiser(model=model, arch=architecture, nature='network', cuda=cuda, sigma=0.01)
            for i, data in enumerate(loader_pnp, 0):
                if i < 2:
                    for n_lev in [noise_lev]:
                        with torch.no_grad(): 
                            data_true = data['image_true'].type(Tensor)  # Keep initial data in memory

                            sol_pnp, blu_pnp, best_pnp = prox_test(data_true, denoiser=denoiser, folder_save=img_folder_pnp, noise_level=n_lev, pth_kernel='blur_models/blur_1.mat', max_iter=1000)  # Takes a batch as input, returns a batch

                            pictures = [sol_pnp, data_true, blu_pnp, best_pnp]
                            pic_paths = [img_folder_pnp+'/Img_sol_'+str(i)+'_'+str(n_lev), img_folder_pnp+'/Img_target_'+str(i)+'_'+str(n_lev), img_folder_pnp+'/Img_blurred_'+str(i)+'_'+str(n_lev), img_folder_pnp+'/Img_best_'+str(i)+'_'+str(n_lev)]
                            save_several_images(pictures, pic_paths, nrows=int(np.sqrt(sol_pnp.shape[0])), out_format='.png')


if __name__ == "__main__":

    train_net(architecture=opt.architecture, n_ch=opt.n_ch, epsilon=opt.epsilon, lambda_jr=opt.lambdajr, noise_lev=opt.noise_level, pth_config_file=opt.pth_config_file)
