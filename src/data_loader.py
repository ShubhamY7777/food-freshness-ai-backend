import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


class FreshnessDataset(Dataset):
    def __init__(self, csv_file=None, root_dir=None, transform=None, dataframe=None):
        """
        You can either pass:
        - csv_file (path)
        OR
        - dataframe directly
        """

        if dataframe is not None:
            self.annotations = dataframe.reset_index(drop=True)
        elif csv_file is not None:
            self.annotations = pd.read_csv(csv_file)
        else:
            raise ValueError("Either csv_file or dataframe must be provided")

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_path = os.path.join(
            self.root_dir,
            self.annotations.loc[idx, "image_path"]
        )

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        days_left = torch.tensor(
            self.annotations.loc[idx, "days_to_spoilage"],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.annotations.loc[idx, "label_state"],
            dtype=torch.long
        )

        return image, days_left, label