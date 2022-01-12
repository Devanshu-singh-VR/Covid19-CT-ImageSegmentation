import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class COVIData(Dataset):
    def __init__(self, image_path, mask_path, transforms):
        self.image = np.load(image_path)
        self.mask = np.load(mask_path).astype(np.uint8)
        self.transforms = transforms

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, item):
        image = self.image[item]
        mask = self.mask[item]

        if self.transforms:
            augmentation = self.transforms(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return image, mask

if __name__ == '__main__':

    dataset = COVIData('pth', 'pth', transforms=None)
    loader = DataLoader(dataset, batch_size=64)
