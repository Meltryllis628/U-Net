import torch
from torchvision import transforms
from PIL import Image



class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, target_paths, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        target = Image.open(self.target_paths[index])

        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)

        return image, target
