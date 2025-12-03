import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_transforms():
    """
    Returns the default transformations for the dataset.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        # Add more transforms here if needed (e.g., normalization)
    ])

class UdacityDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Adjust column indices based on your actual CSV format
        # Usually Udacity dataset has: center, left, right, steering, throttle, brake, speed
        img_name = os.path.join(self.img_dir, os.path.basename(self.data_frame.iloc[idx, 0]))
        
        image = cv2.imread(img_name)
        if image is None:
             # Create a dummy image if file not found for testing purposes
             # In production, you might want to raise an error
             # raise FileNotFoundError(f"Image not found at {img_name}")
             print(f"Warning: Image not found at {img_name}, using black image.")
             image = np.zeros((160, 320, 3), dtype=np.uint8)
             
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        steering_angle = self.data_frame.iloc[idx, 3] # Steering is usually 4th column
        steering_angle = np.array([steering_angle], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, steering_angle

if __name__ == "__main__":
    print("Dataset module loaded successfully.")
    print("UdacityDataset class is ready to be used.")
