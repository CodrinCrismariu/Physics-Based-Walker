import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Tuple

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher

from depth_image_msg import DepthImage_, WIDTH, HEIGHT

TOPIC = "rt/body_depth_camera"
PUBLISH_HZ  = 30
RS_WIDTH    = 640
RS_HEIGHT   = 480
RS_FPS      = 30
DEPTH_MAX_M = 10.0
DEPTH_MIN_M = 0.1

def init_realsense() -> Tuple[rs.pipeline, float]:

    pipeline = rs.pipeline()
    config   = rs.config()
    
    config.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
    profile  = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()
    print(f"[RealSense] Depth scale: {depth_scale} m/unit",
          f"(stream: {RS_WIDTH}x{RS_HEIGHT}) @ {RS_FPS}fps")

    return pipeline, depth_scale

def get_depth_meters(pipeline: rs.pipeline, depth_scale: float):

    frame = pipeline.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    if not depth_frame:
        return None

    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)*depth_scale

    depth[(depth < DEPTH_MIN_M) | (depth > DEPTH_MAX_M)] = 0.0
    return depth

def resize_depth(depth: np.ndarray) -> np.ndarray:
    return cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

def clean_depth(depth: np.ndarray) -> np.ndarray:
    invalid = (depth == 0).astype(np.uint8)

    depth_16 = (depth * 1000).astype(np.uint16)
    filled = cv2.inpaint(
        depth_16.astype(np.float32),
        invalid,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )
    return (filled / 1000.0).astype(np.float32)

def build_message(depth_small: np.ndarray) -> DepthImage_:
    msg              = DepthImage_()
    msg.timestamp_us = int(time.time() * 1e6)
    msg.width        = WIDTH
    msg.height       = HEIGHT
    msg.depth_data   = depth_small.flatten().tolist()
    return msg

def main():
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    pipeline, depth_scale = init_realsense()

    pub = ChannelPublisher(TOPIC, DepthImage_)
    pub.Init()

    print(f"Publishing {WIDTH}x{HEIGHT} depth frames on {TOPIC} at {PUBLISH_HZ} Hz")

    period = 1.0 / PUBLISH_HZ

    try:
        while True:
            loop_start = time.time()

            depth_full = get_depth_meters(pipeline, depth_scale)
            if depth_full is None:
                print("[WARNING] No depth frame received, skipping")
                time.sleep(period)
                continue

            depth_small = resize_depth(depth_full)
            depth_small = clean_depth(depth_small)
            msg         = build_message(depth_small)
            pub.Write(msg)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()

