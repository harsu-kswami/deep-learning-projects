import rclpy
from rclpy.node import Node
import torch
from model_lstm_v2 import ConvNeXtLSTMV2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import cv2
import numpy as np
from cv_bridge import CvBridge

class SteeringNode(Node):
    def __init__(self):
        super().__init__('steering_node')
        self.bridge = CvBridge()

        self.model = ConvNeXtLSTMV2(seq_len=15, hidden=512, dropout=0.3)
        self.model.load_state_dict(torch.load("../outputs/checkpoints/convnext_lstm_v2.pth"))
        self.model.eval().cuda()

        self.buffer = []
        self.seq_len = 15

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )

        self.pub_steer = self.create_publisher(Float32, '/autodrive/steering', 10)
        self.pub_speed = self.create_publisher(Float32, '/autodrive/speed', 10)

    def preprocess(self, img):
        img = cv2.resize(img, (224,224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        return torch.tensor(img)

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = self.preprocess(frame)
        self.buffer.append(frame)

        if len(self.buffer) < self.seq_len:
            return

        seq = torch.stack(self.buffer[-self.seq_len:])
        seq = seq.unsqueeze(0).cuda()

        with torch.no_grad():
            steer, speed = self.model(seq)

        self.pub_steer.publish(Float32(float(steer.item())))
        self.pub_speed.publish(Float32(float(speed.item())))

def main():
    rclpy.init()
    node = SteeringNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
