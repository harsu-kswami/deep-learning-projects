import os
import subprocess
import sys

def optimize_tensorrt():
    onnx_path = "../outputs/onnx/convnext_lstm_v2.onnx"
    engine_path = "../outputs/checkpoints/convnext_lstm_v2.engine"

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        print("Please run onnx_export.py first.")
        return

    # Command to run trtexec
    # --fp16 enables mixed precision for faster inference on Tensor Cores
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]

    print("Running TensorRT optimization...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccess! TensorRT engine saved to {engine_path}")
    except FileNotFoundError:
        print("\nError: 'trtexec' command not found.")
        print("Make sure TensorRT is installed and 'trtexec' is in your system PATH.")
        print("You can download TensorRT from: https://developer.nvidia.com/tensorrt")
    except subprocess.CalledProcessError as e:
        print(f"\nError: trtexec failed with exit code {e.returncode}")

if __name__ == "__main__":
    optimize_tensorrt()
