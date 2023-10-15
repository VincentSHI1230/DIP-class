import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'images').replace('\\', '/')
        mask_path = os.path.join(root, 'labels').replace('\\', '/')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it).replace('\\', '/'), os.path.join(mask_path, it).replace('\\', '/'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'images').replace('\\', '/')
        mask_path = os.path.join(root, 'labels').replace('\\', '/')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it).replace('\\', '/'), os.path.join(mask_path, it).replace('\\', '/'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'images').replace('\\', '/')
        mask_path = os.path.join(root, 'labels').replace('\\', '/')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'test.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it).replace('\\', '/'), os.path.join(mask_path, it).replace('\\', '/'))
            items.append(item)
    return items


class Dataset(data.Dataset):
    def __init__(self, root, mode,crop=None, transform=None):
        self.imgs = make_dataset(root, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.crop = crop

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        file_name = mask_path.split('/')[-1]

        img = cv2.imread(img_path,0)
        mask = cv2.imread(mask_path,0)
        if self.transform is not None:
            img = self.transform(img.copy())
            mask = self.transform(mask)
        return (img, mask), file_name

    def __len__(self):
        return len(self.imgs)
