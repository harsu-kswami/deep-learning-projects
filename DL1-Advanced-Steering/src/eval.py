import torch
from torch.utils.data import DataLoader
from dataset import UdacityDataset, get_transforms
from model_cnn import ConvNeXtTinySteering
import matplotlib.pyplot as plt
import numpy as np

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    ds = UdacityDataset("../data/driving_log.csv",
                        "../data/IMG",
                        transform=get_transforms())
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    # Load trained model
    model = ConvNeXtTinySteering().to(device)
    model.load_state_dict(torch.load("../outputs/checkpoints/convnext_single.pth"))
    model.eval()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for imgs, targets in dl:
            imgs = imgs.to(device)
            out = model(imgs).squeeze(1).cpu().numpy()
            preds_list.extend(out)
            targets_list.extend(targets[:,0].numpy())

    preds = np.array(preds_list)
    y = np.array(targets_list)

    # MAE & RMSE
    mae = np.mean(np.abs(preds - y))
    rmse = np.sqrt(np.mean((preds - y)**2))

    print("MAE:", mae)
    print("RMSE:", rmse)

    # Plot predicted vs actual
    plt.figure(figsize=(10,4))
    plt.plot(y[:300], label="Actual", color="green")
    plt.plot(preds[:300], label="Predicted", color="red")
    plt.legend()
    plt.title("Steering: Actual vs Predicted")
    plt.savefig("../outputs/plots/steering_curve.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
