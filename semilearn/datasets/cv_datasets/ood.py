import os
import torchvision
import numpy as np

from torchvision import transforms
from .datasetbase import BasicDataset

mean, std = {}, {}
mean['svhn'] = [0.4380, 0.4440, 0.4730]
std['svhn'] = [0.1751, 0.1771, 0.1744]
img_size = 32


from torchvision import datasets, transforms
def get_ood(dataset, id, data_dir):
    image_size = (32, 32, 3) if image_size is None else image_size
    DATA_PATH = os.path.join(data_dir, 'ood_data')
    if id == "cifar10":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size = 32
    elif id == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        image_size = 32
    elif "imagenet" in id or id == "tiny":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_size = 224
    test_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),])
    if dataset == 'cifar10':
        test_set = datasets.CIFAR10(data_dir, train=False, download=False, transform=test_transform)
    elif dataset == 'cifar100':
        test_set = datasets.CIFAR100(data_dir, train=False, download=False, transform=test_transform)
    elif dataset == 'svhn':
        test_set = datasets.SVHN(DATA_PATH, split='test', download=True, transform=test_transform)
    elif dataset == 'lsun':
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'imagenet':
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'stanford_dogs':
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'cub':
        test_dir = os.path.join(DATA_PATH, 'cub')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'flowers102':
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'caltech_256':
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'dtd':
        test_dir = os.path.join(DATA_PATH, 'dtd')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    return test_set


def svhn_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['svhn'], std['svhn'])
    ])

    data_dir = os.path.join(data_dir, 'ood_data/svhn')
    dset = torchvision.datasets.SVHN(data_dir, split='test', download=False)
    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels  # data converted to [H, W, C] for PIL
    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 10, transform_val, False, None, False)

    return eval_dset


def lsun_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

    data_dir = os.path.join(data_dir, 'ood_data')
    data = np.load(os.path.join(data_dir, 'LSUN_resize.npy'))
    targets = np.zeros(data.shape[0], dtype=int)

    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 1, transform_val, False, None, False)

    return eval_dset


def gaussian_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

    data_dir = os.path.join(data_dir, 'ood_data')
    data = np.load(os.path.join(data_dir, 'Gaussian.npy'))
    targets = np.zeros(data.shape[0], dtype=int)

    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 1, transform_val, False, None, False)

    return eval_dset


def uniform_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

    data_dir = os.path.join(data_dir, 'ood_data')
    data = np.load(os.path.join(data_dir, 'Uniform.npy'))
    targets = np.zeros(data.shape[0], dtype=int)

    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 1, transform_val, False, None, False)

    return eval_dset
