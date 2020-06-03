import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, RandomRotation, Normalize, Compose

from dataset import ImagenetDataset


def main():

    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_dir = "./tiny-imagenet-200"
    # mean = 112.4660
    # std = 70.8325
    train_transforms = Compose([Normalize(112.4660, 70.8325), RandomRotation(30), ToTensor()])
    test_transforms = ToTensor()

    train_loader = DataLoader(ImagenetDataset(main_dir, 'train', train_transforms), batch_size=64, shuffle=True)
    val_loader = DataLoader(ImagenetDataset(main_dir, 'val', test_transforms), batch_size=64, shuffle=False)
    test_loader = DataLoader(ImagenetDataset(main_dir, 'test', test_transforms), batch_size=64, shuffle=False)

    


if __name__ == '__main__':
    main()
