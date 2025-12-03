from dataset import UdacityDataset, get_transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

# Load dataset
dataset = UdacityDataset(
    csv_path="../data/driving_log.csv",
    img_dir="../data/IMG",
    transform=get_transforms()
)

# Create dataloader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get one batch
images, targets = next(iter(loader))

print("Image batch shape:", images.shape)
print("Target batch shape:", targets.shape)
print("Sample targets:", targets)

# Show first image
img = images[0].permute(1, 2, 0).cpu().numpy()
plt.imshow(img)
plt.title(f"Steering: {targets[0][0].item():.3f}")
plt.show()
