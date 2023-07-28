import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from einops import rearrange
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image

warnings.simplefilter('ignore', category=VerifyWarning)
import warnings

import numpy as np
import torch
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
from torch.utils.data import DataLoader

warnings.simplefilter('ignore', category=VerifyWarning)


CLASSES = ['background', 'spurious', 'compact', 'extended']
COLORS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]


def get_transforms(img_size):
    return  T.Compose([
                RemoveNaNs(),
                ZScale(),
                SigmaClip(),
                ToTensor(),
                torch.nn.Tanh(),
                MinMaxNormalize(),
                Unsqueeze(),
                T.Resize((img_size, img_size)),
                RepeatChannels((3))
            ])

class RemoveNaNs(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img[np.isnan(img)] = 0
        return img


class ZScale(object):
    def __init__(self, contrast=0.15):
        self.contrast = contrast

    def __call__(self, img):
        interval = ZScaleInterval(contrast=self.contrast)
        min, max = interval.get_limits(img)

        img = (img - min) / (max - min)
        return img


class SigmaClip(object):
    def __init__(self, sigma=3, masked=True):
        self.sigma = sigma
        self.masked = masked

    def __call__(self, img):
        img = sigma_clip(img, sigma=self.sigma, masked=self.masked)
        return img


class MinMaxNormalize(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = (img - img.min()) / (img.max() - img.min())
        return img


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32)

class RepeatChannels(object):
    def __init__(self, ch):
        self.ch = ch

    def __call__(self, img):
        return img.repeat(1, self.ch, 1, 1)

class FromNumpy(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.from_numpy(img.astype(np.float32)).type(torch.float32)

class Unsqueeze(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.unsqueeze(0)


def mask_to_rgb(mask):
    rgb_mask = torch.zeros_like(mask, device=mask.device).repeat(1, 3, 1, 1)
    for i, c in enumerate(COLORS):
        color_mask = torch.tensor(c, device=mask.device).unsqueeze(
            1).unsqueeze(2) * (mask == i)
        rgb_mask += color_mask
    return rgb_mask

def get_data_loader(dataset, batch_size, split="train"):
    batch_size = batch_size
    workers = min(8, batch_size)
    is_train = split == "train"
    return DataLoader(dataset, shuffle=is_train, batch_size=batch_size,
                      num_workers=workers, persistent_workers=True,
                      drop_last=is_train
                      )

def rgb_to_tensor(mask):
    r,g,b = mask
    r *= 1
    g *= 2
    b *= 3
    mask, _ = torch.max(torch.stack([r,g,b]), dim=0, keepdim=True)
    return mask


def rand_horizontal_flip(img, mask):
    if random.random() < 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    return img, mask


class RGDataset(Dataset):
    def __init__(self, data_dir, img_paths, img_size=128):
        super().__init__()
        data_dir = Path(data_dir)
        with open(img_paths) as f:
            self.img_paths = f.read().splitlines()
        self.img_paths = [data_dir / p for p in self.img_paths]

        self.transforms = T.Compose([
            RemoveNaNs(),
            ZScale(),
            SigmaClip(),
            ToTensor(),
            torch.nn.Tanh(),
            MinMaxNormalize(),
            # T.Resize((img_size),
            #          interpolation=T.InterpolationMode.NEAREST),
            Unsqueeze(),
            T.Resize((img_size, img_size)),
            
            RepeatChannels((3))
        ])
        self.img_size = img_size

        self.mask_transforms = T.Compose([
            FromNumpy(),
            Unsqueeze(),
            T.Resize((img_size, img_size),
                     interpolation=T.InterpolationMode.NEAREST),
        ])

    def get_mask(self, img_path, type):
        assert type in ["real", "synthetic"], f"Type {type} not supported"
        if type == "real":
            ann_path = str(img_path).replace(
                'imgs', 'masks').replace('.fits', '.json')
            ann_dir = Path(ann_path).parent
            ann_path = ann_dir / f'mask_{ann_path.split("/")[-1]}'
            with open(ann_path) as j:
                mask_info = json.load(j)

            masks = []

            for obj in mask_info['objs']:
                seg_path = ann_dir / obj['mask']

                mask = fits.getdata(seg_path)

                mask = self.mask_transforms(mask.astype(np.float32))
                masks.append(mask)
            mask, _ = torch.max(torch.stack(masks), dim=0)

        elif type == "synthetic":
            mask_path = str(img_path).replace("gen_fits", "cond_fits")
            mask = fits.getdata(mask_path)
            mask = self.mask_transforms(mask)
            mask = mask.squeeze()
            if mask.shape[0] == 3:
                mask = rgb_to_tensor(mask)
        return mask


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        img = fits.getdata(image_path)
        img = self.transforms(img)
        
        if "synthetic" in str(image_path):
            mask = self.get_mask(image_path, type='synthetic')
        else:
            mask = self.get_mask(image_path, type='real')

        # ann_path = str(image_path).replace(
        #     'imgs', 'masks').replace('.fits', '.json')
        # ann_dir = Path(ann_path).parent
        # ann_path = ann_dir / f'mask_{ann_path.split("/")[-1]}'
        # with open(ann_path) as j:
        #     mask_info = json.load(j)


        # masks = []

        # for obj in mask_info['objs']:
        #     seg_path = ann_dir / obj['mask']

        #     mask = fits.getdata(seg_path)

        #     mask = self.mask_transforms(mask.astype(np.float32))
            # masks.append(mask)

        # if 'bkg' in str(image_path):
        #     mask = torch.zeros_like(img)
        #     masks.append(mask)

        # mask, _ = torch.max(torch.stack(masks), dim=0)
        mask = mask.long()
        return img.squeeze(), mask.squeeze()


class SyntheticRGDataset(Dataset):
    def __init__(self, data_dir, img_paths, img_size=128):
        super().__init__()
        data_dir = Path(data_dir)
        with open(img_paths) as f:
            self.img_paths = f.read().splitlines()
        self.img_paths = [data_dir / p for p in self.img_paths]



        self.transforms = T.Compose([
            RemoveNaNs(),
            ZScale(),
            SigmaClip(),
            ToTensor(),
            torch.nn.Tanh(),
            MinMaxNormalize(),
            # T.Resize((img_size),
            #          interpolation=T.InterpolationMode.NEAREST),
            Unsqueeze(),
            T.Resize((img_size, img_size)),
            
            RepeatChannels((3))
        ])
        self.img_size = img_size

        self.mask_transforms = T.Compose([
            FromNumpy(),
            Unsqueeze(),
            T.Resize((img_size, img_size),
                     interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        img = fits.getdata(image_path)
        img = self.transforms(img)
        img = img.squeeze()

        mask_path = str(image_path).replace("gen_fits", "cond_fits")
        mask = fits.getdata(mask_path)
        mask = self.mask_transforms(mask)

        img, mask = rand_horizontal_flip(img, mask)

        mask = mask.squeeze().long()
        return img, mask


if __name__ == '__main__':
    rgtrain = SyntheticRGDataset('data/rg-dataset/data',
                        'data/rg-dataset/val_w_bg.txt')
    batch = next(iter(rgtrain))
    image, mask, masked_image = batch
    to_pil_image(image).save('image.png')
    rgb_mask = mask_to_rgb(mask)[0]
    to_pil_image(rgb_mask).save('mask.png')
    to_pil_image(masked_image[0]).save('masked.png')

    bs = 256

    loader = torch.utils.data.DataLoader(
        rgtrain, batch_size=bs, shuffle=False, num_workers=16)
    for i, batch in enumerate(loader):
        image, mask, masked_image = batch
        rgb_mask = mask_to_rgb(mask)
        nrow = int(math.sqrt(bs))
        # nrow = bs // 2
        grid = make_grid(rgb_mask, nrow=nrow, padding=0)
        save_image(grid, f'mask_{nrow}x{nrow}.png')
        break