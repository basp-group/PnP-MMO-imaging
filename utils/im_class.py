
import glob
import os.path as osp

import numpy as np
from PIL import Image

import torch
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms


class imnetDataset(D.Dataset):
    """
    Loads the ImageNet dataset
    """
    def __init__(self, path_data, transform=None, reduction_size=None, pattern='*.JPEG', channels=3):
        self.transform = transform
        self.channels = channels
        if reduction_size is None:
            self.filenames = glob.glob(osp.join(path_data, pattern))
        else:
            filenames_list = sorted(glob.glob(osp.join(path_data, pattern)))   
            self.filenames = filenames_list[:reduction_size]
        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ 
        Get a sample from the dataset
        """
        image_path = self.filenames[index]
        image_name = osp.basename(image_path)
        image_true = Image.open(image_path)
        image_true = np.asarray(image_true, dtype="float32")

        if len(image_true.shape) < 3:
            image_true = np.repeat(image_true[:, :, np.newaxis], 3, axis=2)
        else:
            if image_true.shape[2] > 3:  # Strange case that seems to appear at least once
                image_true = image_true[:, :, :3]

        if self.transform is not None:
            image_true = Image.fromarray(np.uint8(image_true))
            image_true = self.transform(image_true)
            image_true = np.asarray(image_true, dtype="float32")

        if self.channels == 1:
            image_true = 0.2125*image_true[..., 0:1]+0.7154*image_true[..., 1:2]+0.0721*image_true[..., 2:3]

        image_true = torch.from_numpy(np.moveaxis((image_true/255.), -1, 0))
 
        return {'image_true': image_true, 'image_name': image_name}

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_dataset(path_dataset='/datasets/ImageNet/', path_red_dataset='/datasets/BSDS300/', random_seed=30, bs=50,
                patchSize=64, red=False, red_size=25, channels=3, pattern_red='*.JPEG', num_workers=6):
    """
    Returns the dataloader for the dataset
    """

    data_transform_true = transforms.Compose([
        transforms.RandomResizedCrop(patchSize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    img_dataset = imnetDataset(path_data=path_dataset, transform=data_transform_true, reduction_size=None,
                               channels=channels)
    
    validation_split = .02
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(img_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    torch.manual_seed(random_seed)

    # Creating samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader_train = torch.utils.data.DataLoader(img_dataset, batch_size=bs, 
                                               sampler=train_sampler, num_workers=num_workers)
    loader_val = torch.utils.data.DataLoader(img_dataset, batch_size=bs,
                                             sampler=valid_sampler, num_workers=num_workers)

    if red:
        img_dataset_reduced = imnetDataset(path_data=path_red_dataset, transform=None, reduction_size=red_size,
                                           channels=channels, pattern=pattern_red)
        loader_red = torch.utils.data.DataLoader(img_dataset_reduced, batch_size=1)
        return loader_train, loader_val, loader_red
    else:
        return loader_train, loader_val


def get_dataset_test(path_dataset='../datasets/BSDS300/', red_size=25, channels=3, pattern_red='*/*.JPEG'):
    """
    Returns the dataloader for the dataset
    """

    img_dataset = imnetDataset(path_data=path_dataset, transform=None, reduction_size=red_size, channels=channels,
                               pattern=pattern_red)
    loader_data = torch.utils.data.DataLoader(img_dataset, batch_size=1)

    return loader_data
