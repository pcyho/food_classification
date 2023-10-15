from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from torchvision.transforms.functional import resize


class Data(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_path = os.listdir(data_dir)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img = read_image(os.path.join(self.data_dir, img_name))
        img = resize(img, (128, 128)) 
        label = int(img_name.split("_")[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_path)
