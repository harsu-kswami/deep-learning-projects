import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from augment import augment

class UdacitySequenceDatasetV2(Dataset):
    def __init__(self, csv_path, img_dir, seq_len=15, use_aug=True):
        self.data = pd.read_csv(csv_path, header=None)
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.use_aug = use_aug

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data) - self.seq_len

    def fix_path(self, path):
        return os.path.join(self.img_dir, os.path.basename(path))

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        seq_imgs = []
        seq_steer = []

        for i in range(self.seq_len):
            row = self.data.iloc[idx + i]
            img_path = self.fix_path(row[0])
            steering = float(row[3])

            img = self.load_img(img_path)

            if self.use_aug:
                img, steering = augment(img, steering)

            seq_imgs.append(self.transform(img))
            seq_steer.append(steering)

        seq_imgs = torch.stack(seq_imgs)  # [T, C, H, W]

        # Target steering = steering at last frame
        steer = torch.tensor([seq_steer[-1]], dtype=torch.float32)

        # Synthetic speed target (smooth curve)
        # can replace with real speed if available
        speed = torch.tensor([len(seq_steer) * 0.8], dtype=torch.float32)

        return seq_imgs, torch.cat([steer, speed], dim=0)
