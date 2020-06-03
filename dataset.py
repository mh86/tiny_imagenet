import torch
from torch.utils.data import Dataset
from os import scandir
import numpy as np
import cv2
import pandas as pd


def create_label_dict(main_dir):
    dic_ind = {}
    with open(main_dir + '/wnids.txt', 'r') as f:
        for i, k in enumerate(f.readlines()):
            dic_ind[k.rstrip('\n')] = i
    return dic_ind


def create_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    if len(image.shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[0] != 64:
        image = cv2.resize(image, (64, 64, 3))
    return image


class ImagenetDataset(Dataset):

    def __init__(self, main_dir, mode='train', transforms=None):
        super(ImagenetDataset, self).__init__()
        self.mode = mode
        self.transforms = transforms
        self.label_ind = create_label_dict(main_dir)
        self.images, self.labels = self.load_images(main_dir)

    def __getitem__(self, index):
        image = self.transforms(self.images[index]) if self.transforms else \
            torch.from_numpy(self.images[index])
        if self.mode != 'test':
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return self.images.size()[0]

    def get_image_paths(self, main_dir):
        if self.mode == 'train':
            image_list = []
            subfolders = [f.path for f in scandir(main_dir + '/' + self.mode) if f.is_dir()]
            for path in subfolders:
                image_list.extend([f.path for f in scandir(path + "/images") if f.is_file()])
        else:
            image_list = [f.path for f in scandir(main_dir + '/' + self.mode + "/images") if f.is_file()]
        return len(image_list), image_list

    def load_images(self, main_dir):
        n, image_list = self.get_image_paths(main_dir)
        images = np.zeros((n, 64, 64, 3))
        if self.mode == 'train':
            labels = [0] * n
        for i, image_path in enumerate(image_list):
            images[i] = create_image(image_path)
            if self.mode == 'train':
                labels[i] = image_path.split('/')[-3]
        if self.mode == 'val':
            labels = pd.read_csv(main_dir + '/' + self.mode +
                                          '/val_annotations.txt', sep='\t', header=None)[1].values

        if self.mode != 'test':
            labels = np.array([self.label_to_ind(label) for label in labels])
            return images, labels
        else:
            return images

    def label_to_ind(self, label):
        return self.label_ind[label]


# # Sanity check
# imagenet = ImagenetDataset("./tiny-imagenet-200")
# mean = 112.4660
# std = 70.8325
# img, label = imagenet.__getitem__(2)
# print(img, label)
