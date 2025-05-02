
import copy
import os

import numpy as np
from PIL import Image
from torchvision import transforms
import math

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from .datasetbase import BasicDataset

mean, std = {}, {}
mean['semi_inat'] = [0.4732, 0.4828, 0.3779]
std['semi_inat'] = [0.2348, 0.2243, 0.2408]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_semi_inat(args, alg, name, num_labels, num_classes, data_dir='./data'):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['semi_inat'], std['semi_inat'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['semi_inat'], std['semi_inat'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['semi_inat'], std['semi_inat'])
    ])

    # data_dir = os.path.join(data_dir, name.lower())
    data_dir = os.path.join(data_dir, "semi-inat-2021")

    if alg == 'ssb':
        lb_dset = SiNatDataset(root=os.path.join(data_dir, "l_train/l_train"),
                            transform=transform_weak,
                            is_ulb=False,
                            alg=alg,
                            strong_transform=transform_strong,
                            flist=os.path.join(data_dir, f'filelist/train_labeled.txt'))
    else:
        lb_dset = SiNatDataset(root=os.path.join(data_dir, "l_train/l_train"),
                            transform=transform_weak,
                            is_ulb=False,
                            alg=alg,
                            flist=os.path.join(data_dir, f'filelist/train_labeled.txt'))

    ulb_dset = SiNatDataset(root=os.path.join(data_dir, "l_train/l_train"),
                           transform=transform_weak,
                           is_ulb=True,
                           alg=alg,
                           strong_transform=transform_strong,
                           flist=os.path.join(data_dir, f'filelist/train_unlabeled.txt'),
                           mode="unseen")

    test_dset = SiNatDataset(root=os.path.join(data_dir, "l_train/l_train"),
                            transform=transform_val,
                            is_ulb=False,
                            alg=alg,
                            flist=os.path.join(data_dir, f'filelist/test.txt'))

    test_data, test_targets = test_dset.data, test_dset.targets
    test_targets[test_targets >= num_classes] = num_classes
    seen_indices = np.where(test_targets < num_classes)[0]

    eval_dset = copy.deepcopy(test_dset)
    eval_dset.data, eval_dset.targets = eval_dset.data[seen_indices], eval_dset.targets[seen_indices]

    return lb_dset, ulb_dset, eval_dset, test_dset


def make_dataset_from_list(flist, num_classes=0, mode="seen"):
    with open(flist) as f:
        lines = f.readlines()
        imgs = [line.split(' ')[0] for line in lines]
        if mode == "seen":
            targets = [int(line.split(' ')[1].strip()) for line in lines]
        elif mode == "unseen":
            targets = [int(num_classes) for _ in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets


def find_classes(directory):
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class SiNatDataset(BasicDataset):
    def __init__(self, root, transform, is_ulb, alg, strong_transform=None, flist=None, mode="seen"):
        super(SiNatDataset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                          transform=transform, strong_transform=strong_transform)
        self.root = root
        classes, class_to_idx = find_classes(self.root)

        if mode == "seen":
            imgs, targets = make_dataset_from_list(flist)
        if mode == "unseen":
            num_classes = len(classes)
            imgs, targets = make_dataset_from_list(flist, num_classes=num_classes, mode=mode)

        if len(imgs) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = pil_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = imgs
        self.targets = targets

        self.strong_transform = strong_transform

    def __sample__(self, idx):
        path, target = self.data[idx], self.targets[idx]
        img = self.loader(path)
        return img, target
