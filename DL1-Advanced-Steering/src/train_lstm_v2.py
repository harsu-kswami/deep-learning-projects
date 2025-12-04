import torch
from torch.utils.data import DataLoader
from model_lstm_v2 import ConvNeXtLSTMV2
from dataset_lstm_v2 import UdacitySequenceDatasetV2
import torch.nn as nn

def train_lstm_v2():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = UdacitySequenceDatasetV2(
        csv_path="../data/driving_log.csv",
        img_dir="../data/IMG",
        seq_len=15,
        use_aug=True
    )

    dl = DataLoader(
        ds, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True
    )

    model = ConvNeXtLSTMV2(seq_len=15, hidden=512, dropout=0.3).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    loss_steer = nn.SmoothL1Loss()
    loss_speed = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(100):
        total_loss = 0

        for seq, target in dl:
            seq = seq.to(device)
            steer_gt = target[:,0].to(device)
            speed_gt = target[:,1].to(device)

            with torch.cuda.amp.autocast():
                pred_steer, pred_speed = model(seq)

                ls = loss_steer(pred_steer.squeeze(1), steer_gt)
                ls_speed = loss_speed(pred_speed.squeeze(1), speed_gt)

                loss = ls + 0.3 * ls_speed  # weighted multi-task

            opt.zero_grad()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}: {total_loss/len(dl):.4f}")

    torch.save(model.state_dict(), "../outputs/checkpoints/convnext_lstm_v2.pth")
    print("Saved: convnext_lstm_v2.pth")

if __name__ == "__main__":
    train_lstm_v2()
