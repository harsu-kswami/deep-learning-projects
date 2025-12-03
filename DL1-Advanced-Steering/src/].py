# TODO: training pipeline
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UdacityDataset, get_transforms
from model_cnn import ConvNeXtTinySteering

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = UdacityDataset("../data/driving_log.csv",
                        "../data/IMG",
                        transform=get_transforms())
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = ConvNeXtTinySteering().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(5):
        total = 0
        for imgs, targets in dl:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs).squeeze(1)   # [B] from [B,1]
            loss = loss_fn(preds, targets[:,0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Epoch {epoch+1}: Loss={total/len(dl):.4f}")

    torch.save(model.state_dict(), "../outputs/checkpoints/convnext_single.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
