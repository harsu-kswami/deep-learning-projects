import cv2, torch, time
import numpy as np
from model_lstm_v2 import ConvNeXtLSTMV2
from dataset_lstm_v2 import UdacitySequenceDatasetV2
from pid import EMASmoother

def video_advanced():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = UdacitySequenceDatasetV2(
        csv_path="../data/driving_log.csv",
        img_dir="../data/IMG",
        seq_len=15,
        use_aug=False
    )

    model = ConvNeXtLSTMV2(seq_len=15, hidden=512, dropout=0.3).to(device)
    model.load_state_dict(torch.load("../outputs/checkpoints/convnext_lstm_v2.pth"))
    model.eval()

    smoother = EMASmoother(alpha=0.25)

    out = cv2.VideoWriter(
        "../outputs/demo_videos/lstm_v2_demo.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (320, 160)
    )

    buffer = []
    prev_time = time.time()

    for i in range(len(ds)):
        seq, target = ds[i]

        # we want last frame in sequence for visualization
        img_path = ds.fix_path(ds.data.iloc[i + ds.seq_len - 1][0])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (320,160))

        buffer.append(seq[-1])
        if len(buffer) < ds.seq_len:
            out.write(img)
            continue

        x = torch.stack(buffer[-ds.seq_len:]).unsqueeze(0).to(device)

        with torch.no_grad():
            steer_pred, speed_pred = model(x)
            steer_pred = steer_pred.item()
            speed_pred = speed_pred.item()

        steer_gt = target[0].item()
        steer_smooth = smoother.update(steer_pred)

        # FPS calc
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # HUD overlay
        cv2.putText(img, f"GT: {steer_gt:.2f}", (10,20), 0, 0.55, (0,255,0), 2)
        cv2.putText(img, f"P: {steer_pred:.2f}", (10,45), 0, 0.55, (0,0,255), 2)
        cv2.putText(img, f"S: {steer_smooth:.2f}", (10,70), 0, 0.55, (255,0,0), 2)
        cv2.putText(img, f"SPD: {speed_pred:.1f}", (10,95), 0, 0.55, (255,255,0), 2)
        cv2.putText(img, f"FPS: {fps:.1f}", (10,120), 0, 0.55, (255,255,255), 2)

        out.write(img)

    out.release()
    print("Saved video: lstm_v2_demo.mp4")

if __name__ == "__main__":
    video_advanced()
