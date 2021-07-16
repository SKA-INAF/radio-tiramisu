import os
from utils.training import test, train
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from datasets import joint_transforms
from torchvision.datasets.folder import is_image_file, default_loader
import torchvision.transforms as transforms

classes = ['Void', 'Sidelobe', 'Source', 'Galaxy']

class_color = [
    (0, 0, 0),    
    (32,207,227),
    (250,7,7),
    (237,237,12),
]


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class LabelToLongTensor(object):
    def __call__(self, pic):        
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().long()
        return label


class LabelTensorToPILImage(object):
    def __call__(self, label):
        label = label.unsqueeze(0)
        colored_label = torch.zeros(3, label.size(1), label.size(2)).byte()
        for i, color in enumerate(class_color):
            mask = label.eq(i)
            for j in range(3):
                colored_label[j].masked_fill_(mask, color[j])
        npimg = colored_label.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        mode = None
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]
            mode = "L"

        return Image.fromarray(npimg, mode=mode)


class AstroDataset(data.Dataset):

    def __init__(self, root, split='train', joint_transform=None,
                 transform=None, target_transform=None,
                 loader=default_loader):
        self.root = root
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.classes = classes
        self.mean = [0.28535324, 0.28535324, 0.28535324]
        self.std = [0.28536762, 0.28536762, 0.28536762]

        self.imgs = _make_dataset(os.path.join(self.root, self.split))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        target = Image.open(path.replace(self.split, self.split + 'annot'))

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])

        if self.transform is not None:
            img = self.transform(img)
        
        target = self.target_transform(target)        
        return img, target

    def __len__(self):
        return len(self.imgs)


class AstroDataLoaders():
    def __init__(self, data_path, batch_size):
        self.mean = [0.28535324, 0.28535324, 0.28535324]
        self.std = [0.28536762, 0.28536762, 0.28536762]
        self.data_path = data_path
        self.batch_size = batch_size
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.class_weight = torch.FloatTensor([0.25, 2.85, 0.30, 1.50])


    def get_train_loader(self):
        train_joint_transformer = transforms.Compose([
            #joint_transforms.JointRandomCrop(224), # commented for fine-tuning
            joint_transforms.JointRandomHorizontalFlip()
            ])
        train_dset = AstroDataset(self.data_path, 'train',
            joint_transform=train_joint_transformer,
            transform=transforms.Compose([
                transforms.Resize([132, 132]),
                transforms.ToTensor(),
                self.normalize,
            ]),
            target_transform=transforms.Compose([
                #transforms.Resize([132, 132]),
                LabelToLongTensor(),
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def get_val_loader(self):
        val_dset = AstroDataset(
            self.data_path, 'val', joint_transform=None,
            transform=transforms.Compose([
                transforms.Resize([132, 132]),
                transforms.ToTensor(),
                self.normalize
        ]),
        target_transform=transforms.Compose([
            #transforms.Resize([132, 132]),
            LabelToLongTensor(),
        ]))
        
        val_loader = torch.utils.data.DataLoader(
            val_dset, batch_size=self.batch_size, shuffle=False)

        return val_loader

    def get_test_loader(self):
        test_dset = AstroDataset(
            self.data_path, 'test', joint_transform=None,
            transform=transforms.Compose([
                transforms.Resize([132, 132]),
                transforms.ToTensor(),
                self.normalize
            ]),
            target_transform=transforms.Compose([
                #transforms.Resize([132, 132]),
                LabelToLongTensor(),
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dset, batch_size=self.batch_size, shuffle=False)

        return test_loader