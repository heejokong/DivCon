import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ossl_data, reassign_target


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_cifar_openset(args, alg, name, num_labels, num_classes, data_dir='./data', pure_unlabeled=False):
    name = name.split('_')[0]  # cifar10_openset -> cifar10
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=False)
    data, targets = dset.data, dset.targets

    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name], )
    ])

    if name == 'cifar10':
        num_all_classes = 10
        if args.correlated_ood:
            seen_classes = set(range(0, 6))
        else:
            seen_classes = set(range(2, 8))

    elif name == 'cifar100':
        num_all_classes = 100
        super_classes = np.array([
            [ 4, 30, 55, 72, 95], [ 1, 32, 67, 73, 91], [54, 62, 70, 82, 92], [ 9, 10, 16, 28, 61],
            [ 0, 51, 53, 57, 83], [22, 39, 40, 86, 87], [ 5, 20, 25, 84, 94], [ 6,  7, 14, 18, 24],
            [ 3, 42, 43, 88, 97], [12, 17, 37, 68, 76], [23, 33, 49, 60, 71], [15, 19, 21, 31, 38],
            [34, 63, 64, 66, 75], [26, 45, 77, 79, 99], [ 2, 11, 35, 46, 98], [27, 29, 44, 78, 93],
            [36, 50, 65, 74, 80], [47, 52, 56, 59, 96], [ 8, 13, 48, 58, 90], [41, 69, 81, 85, 89],
            ])
        if args.correlated_ood:
            per_coarse = num_classes // 20
            seen_classes = set(np.unique(super_classes[:,:per_coarse].flatten()))
        else:
            num_coarse = num_classes // 5
            seen_classes = set(np.unique(super_classes[:num_coarse,:].flatten()))

    else:
        raise NotImplementedError

    lb_data, lb_targets, ulb_data, ulb_targets = split_ossl_data(args, data, targets, num_labels, num_all_classes, seen_classes, include_lb_to_ulb=True)
    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    if alg == 'ssb':
        lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)
    else:
        lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    if pure_unlabeled:
        seen_indices = np.where(ulb_targets < num_classes)[0]
        ulb_data = ulb_data[seen_indices]
        ulb_targets = ulb_targets[seen_indices]
    
    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_all_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=False)
    test_data, test_targets = dset.data, reassign_target(dset.targets, num_all_classes, seen_classes)
    seen_indices = np.where(test_targets < num_classes)[0]
    eval_dset = BasicDataset(alg, test_data[seen_indices], test_targets[seen_indices], len(seen_classes), transform_val, False, None, False)
    test_full_dset = BasicDataset(alg, test_data, test_targets, num_all_classes, transform_val, False, None, False)
    return lb_dset, ulb_dset, eval_dset, test_full_dset
