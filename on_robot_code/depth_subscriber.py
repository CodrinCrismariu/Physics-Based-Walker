import sys
import time
import numpy as np
import cv2

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from depth_image_msg import DepthImage_, WIDTH, HEIGHT

TOPIC = "rt/body_depth_camera"
OUTPUT_DIR = "."

def on_depth_message(msg: DepthImage_):
    depth = np.array(msg.depth_data, dtype=np.float32).reshape(HEIGHT, WIDTH)

    display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    display = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)

    filename = f"{OUTPUT_DIR}/depth.png"
    cv2.imwrite(filename, display)
    print(f"Saved {filename}")

def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber(TOPIC, DepthImage_)
    sub.Init(on_depth_message, 10)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()