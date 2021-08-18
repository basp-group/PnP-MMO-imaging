import torch
import torch.nn as nn

import math

from .basic_models import simple_CNN


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)


def get_model(architecture, n_ch=1):

    if architecture == 'DnCNN_nobn':
        net = simple_CNN(n_ch_in=n_ch, n_ch_out=n_ch, n_ch=64, nl_type='relu', depth=20, bn=False)
        clip_val, lr = 1e-2, 1e-4
        net_name = 'dncnn_nobn_nch_'+str(n_ch)
    else:
        raise ValueError('Unknown architecture type.')

    if 'DnCNN' in architecture:
        net.apply(weights_init_kaiming)

    # Move to GPU if possible
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        print("cuda driver found - training on GPU.\n")
        net.cuda()
    else:
        print("no cuda driver found - training on CPU.\n")

    if cuda:
        model = nn.DataParallel(net).cuda()
    else:
        model = nn.DataParallel(net)

    return model, net_name, clip_val, lr
