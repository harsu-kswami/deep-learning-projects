import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model_lstm_v2 import ConvNeXtLSTMV2
from dataset_lstm_v2 import UdacitySequenceDatasetV2
import os

def eval_final():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # output directory
    os.makedirs("../outputs/plots", exist_ok=True)

    # Dataset (NO augmentation)
    ds = UdacitySequenceDatasetV2(
        csv_path="../data/driving_log.csv",
        img_dir="../data/IMG",
        seq_len=15,
        use_aug=False
    )
    dl = DataLoader(ds, batch_size=16, shuffle=False)

    # Model
    model = ConvNeXtLSTMV2(seq_len=15, hidden=512, dropout=0.3).to(device)
    model.load_state_dict(torch.load("../outputs/checkpoints/convnext_lstm_v2.pth"))
    model.eval()

    steer_preds, steer_gt = [], []
    speed_preds, speed_gt = [], []

    # ---------------- INFERENCE LOOP ----------------
    with torch.no_grad():
        for seq, target in dl:
            seq = seq.to(device)
            gt_steer = target[:,0].numpy()
            gt_speed = target[:,1].numpy()

            pred_steer, pred_speed = model(seq)
            pred_steer = pred_steer.squeeze(1).cpu().numpy()
            pred_speed = pred_speed.squeeze(1).cpu().numpy()

            steer_preds.extend(pred_steer)
            steer_gt.extend(gt_steer)
            speed_preds.extend(pred_speed)
            speed_gt.extend(gt_speed)

    steer_preds = np.array(steer_preds)
    steer_gt = np.array(steer_gt)
    speed_preds = np.array(speed_preds)
    speed_gt = np.array(speed_gt)

    # ---------------------------------------------------------------------
    # METRICS: Steering
    mae_steer = np.mean(np.abs(steer_preds - steer_gt))
    rmse_steer = np.sqrt(np.mean((steer_preds - steer_gt)**2))
    max_err_steer = np.max(np.abs(steer_preds - steer_gt))

    # METRICS: Speed
    mae_speed = np.mean(np.abs(speed_preds - speed_gt))
    rmse_speed = np.sqrt(np.mean((speed_preds - speed_gt)**2))

    # Steering smoothness metrics
    jitter = np.mean(np.abs(np.diff(steer_preds)))
    smoothness = 1.0 / (1.0 + jitter)

    print("\n=============== FINAL EVALUATION ===============")
    print(f" Steering MAE:       {mae_steer:.4f}")
    print(f" Steering RMSE:      {rmse_steer:.4f}")
    print(f" Steering Max Error: {max_err_steer:.4f}")
    print("-----------------------------------------------")
    print(f" Speed MAE:          {mae_speed:.4f}")
    print(f" Speed RMSE:         {rmse_speed:.4f}")
    print("-----------------------------------------------")
    print(f" Smoothness:         {smoothness:.4f}")
    print("================================================\n")

    # ------------------ PLOTS ---------------------

    # 1) Steering Curve (300 samples)
    plt.figure(figsize=(10,4))
    plt.plot(steer_gt[:300], label="GT Steering", color="green")
    plt.plot(steer_preds[:300], label="Pred Steering", color="red")
    plt.legend()
    plt.title("Steering: Actual vs Predicted")
    plt.savefig("../outputs/plots/final_curve.png")

    # 2) Error Histogram
    plt.figure(figsize=(6,4))
    plt.hist(steer_preds - steer_gt, bins=40, color="blue", alpha=0.7)
    plt.title("Steering Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.savefig("../outputs/plots/error_hist.png")

    # 3) Steering Distribution Curve
    plt.figure(figsize=(6,4))
    plt.hist(steer_gt, bins=40, color="green", alpha=0.7)
    plt.title("Ground Truth Steering Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Count")
    plt.savefig("../outputs/plots/steering_distribution.png")

    # 4) Speed GT vs Predictions
    plt.figure(figsize=(10,4))
    plt.plot(speed_gt[:300], label="GT Speed", color="orange")
    plt.plot(speed_preds[:300], label="Pred Speed", color="purple")
    plt.legend()
    plt.title("Speed: Actual vs Predicted")
    plt.savefig("../outputs/plots/speed_curve.png")

    print("Saved: final_curve.png, error_hist.png, "
          "steering_distribution.png, speed_curve.png")

if __name__ == "__main__":
    eval_final()
