import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import torch
from model_lstm_v2 import ConvNeXtLSTMV2

def export_onnx():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConvNeXtLSTMV2(seq_len=15, hidden=512, dropout=0.3).to(device)
    model.load_state_dict(torch.load("../outputs/checkpoints/convnext_lstm_v2.pth"))
    model.eval()

    dummy = torch.randn(1, 15, 3, 224, 224).to(device)

    out_path = "../outputs/onnx/convnext_lstm_v2.onnx"
    os.makedirs("../outputs/onnx", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=['frames'],
        output_names=['steer', 'speed'],
        opset_version=17,
        dynamic_axes={
            'frames': {0: 'batch'},
            'steer':  {0: 'batch'},
            'speed':  {0: 'batch'}
        }
    )

    print("ONNX exported →", out_path)

if __name__ == "__main__":
    export_onnx()
