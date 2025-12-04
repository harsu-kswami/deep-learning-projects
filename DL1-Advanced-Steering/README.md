# 🚗 DL-1 Steering System (ConvNeXt + LSTM + Multi-Task Learning)

This project implements an ** Autonomous Steering System** using:

- **ConvNeXt CNN Backbone**
- **LSTM Temporal Modeling**
- **Multi-Task Prediction** (Steering + Speed)
- **ONNX Export for Deployment**
- **Smooth Steering Correction**
- **Full Evaluation (MAE, RMSE, Smoothness)**
- **Video Rendering & HUD Overlay**


---
```text
DL1-Advanced-Steering/
│
├── data/
│   ├── IMG/
│   └── driving_log.csv
│
├── src/
│   ├── dataset_lstm_v2.py        # Sequence dataset
│   ├── model_lstm_v2.py          # ConvNeXt + LSTM model
│   ├── train_lstm_v2.py          # Training script
│   ├── eval_final.py             # Full metrics + plots
│   └── video_advanced.py         # HUD steering demo video
│
├── deployment/
│   ├── onnx_export.py            # Export to ONNX
│   ├── tensorrt_opt.py           # TensorRT engine
│   └── ros2_node.py              # ROS2 deployment node
│
└── outputs/
    ├── checkpoints/
    │   └── convnext_lstm_v2.pth
    ├── plots/
    │   ├── final_curve.png
    │   ├── error_hist.png
    │   ├── steering_distribution.png
    │   └── speed_curve.png
    └── demo_videos/
        └── lstm_demo.mp4
```

# 🎯 Model Capabilities

### ✔ Predicts Steering Angle (float)  
### ✔ Predicts Vehicle Speed (multi-task)  
### ✔ Uses **15-frame temporal sequence**  
### ✔ ConvNeXt backbone improves perception  
### ✔ LSTM stabilizes predictions  
### ✔ Perfect for robotics + AI resume  

---

# 🧠 Training Summary

| Metric | Value |
|--------|--------|
| **Steering MAE** | 0.0164 |
| **Steering RMSE** | 0.0221 |
| **Speed MAE** | 0.0040 |
| **Speed RMSE** | 0.0050 |
| **Smoothness** | 0.8577 |

High-quality performance for behavioral cloning.

---

# 📉 Visual Outputs (Upload Your Files Here)

### **1️⃣ Steering: Actual vs Predicted**
📌 Add ``

### **2️⃣ Error Histogram**
📌 Add `error_hist.png`

### **3️⃣ Steering Distribution**
📌 Add `steering_distribution.png`

### **4️⃣ Speed Prediction Curve**
📌 Add `speed_curve.png`

### **5️⃣ Video Demonstration**
Upload your video file:

outputs/demo_videos/lstm_demo.mp4

kotlin
Copy code

Embed like this:

https://github.com/YOUR_USERNAME/YOUR_REPO/raw/main/outputs/demo_videos/lstm_demo.mp4

yaml
Copy code

---

# 🤖 ONNX Export (Deployment Ready)

Export ONNX:

```bash
python deployment/onnx_export.py
This generates:

Copy code
model_advanced.onnx
ONNX Benefits
Can run on C++, Unity, Web, Python

Works with ONNX Runtime

Required for TensorRT

Faster + portable

Ideal for robotics, embedded systems, real-time inference

🚀 How to Run the Project
1️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
2️⃣ Train Model
bash
Copy code
python src/train_lstm_v2.py
3️⃣ Evaluate (plots + metrics)
bash
Copy code
python src/eval_final.py
4️⃣ Generate Autopilot Demo Video
bash
Copy code
python src/video_advanced.py
5️⃣ Export ONNX Model
bash
Copy code
python deployment/onnx_export.py
