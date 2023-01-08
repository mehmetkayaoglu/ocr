import os
import torch
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.img_dir = root
        self.image_names = os.listdir(root)
        self.img_labels = [os.path.splitext(filename)[0] for filename in self.image_names]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_names[index])
        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)

        label = self.img_labels[index]
        label = label.encode('utf-8')
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


def collate(batch):
    images, labels = zip(*batch)
    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    return images, labels
