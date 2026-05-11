#!/usr/bin/env python3
"""
realsense_node.py
─────────────────
Streams Intel RealSense D435 RGB + depth + camera_info over ROS2.

Topics published:
  /camera/color/image_raw          sensor_msgs/Image        (BGR8)
  /camera/color/camera_info        sensor_msgs/CameraInfo
  /camera/depth/image_rect_raw     sensor_msgs/Image        (16UC1 mm)
  /camera/depth/image_raw_float    sensor_msgs/Image        (32FC1 m)
  /camera/aligned_depth/image_raw  sensor_msgs/Image        (depth aligned to color, 32FC1 m)
"""

import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge


class RealSenseNode(Node):

    def __init__(self):
        super().__init__("realsense_node")

        self.declare_parameter("width",      640)
        self.declare_parameter("height",     480)
        self.declare_parameter("fps",        30)
        self.declare_parameter("frame_id",   "camera_color_optical_frame")
        self.declare_parameter("depth_frame_id", "camera_depth_optical_frame")

        self.W   = self.get_parameter("width").value
        self.H   = self.get_parameter("height").value
        self.FPS = self.get_parameter("fps").value
        self.frame_id       = self.get_parameter("frame_id").value
        self.depth_frame_id = self.get_parameter("depth_frame_id").value

        self.bridge = CvBridge()

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.pub_rgb       = self.create_publisher(Image, "/camera/color/image_raw",         best_effort)
        self.pub_info      = self.create_publisher(CameraInfo, "/camera/color/camera_info",  best_effort)
        self.pub_depth_mm  = self.create_publisher(Image, "/camera/depth/image_rect_raw",    best_effort)
        self.pub_depth_m   = self.create_publisher(Image, "/camera/aligned_depth/image_raw", best_effort)

        self._init_pipeline()
        self.create_timer(1.0 / self.FPS, self._grab)
        self.get_logger().info(
            f"RealSense D435 streaming {self.W}x{self.H} @ {self.FPS} fps"
        )

    def _init_pipeline(self):
        self.pipe   = rs.pipeline()
        cfg         = rs.config()
        cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8,   self.FPS)
        cfg.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16,    self.FPS)

        try:
            profile = self.pipe.start(cfg)
        except RuntimeError as e:
            self.get_logger().error(f"RealSense no conectada: {e}")
            self.get_logger().error("Conecta la cámara y reinicia el nodo.")
            raise

        # Align depth → color frame
        self.align = rs.align(rs.stream.color)

        # Intrinsics for camera_info
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self._camera_info = self._build_camera_info(intr)

        # Depth scale (device units → metres)
        depth_sensor    = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()   # typically 0.001

    def _build_camera_info(self, intr) -> CameraInfo:
        ci = CameraInfo()
        ci.width  = intr.width
        ci.height = intr.height
        ci.distortion_model = "plumb_bob"
        ci.d  = list(intr.coeffs)
        ci.k  = [intr.fx, 0, intr.ppx,
                 0, intr.fy, intr.ppy,
                 0, 0, 1]
        ci.r  = [1, 0, 0,  0, 1, 0,  0, 0, 1]
        ci.p  = [intr.fx, 0, intr.ppx, 0,
                 0, intr.fy, intr.ppy, 0,
                 0, 0, 1, 0]
        return ci

    def _grab(self):
        frames = self.pipe.wait_for_frames(timeout_ms=100)
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        now = self.get_clock().now().to_msg()
        header_rgb   = Header(stamp=now, frame_id=self.frame_id)
        header_depth = Header(stamp=now, frame_id=self.frame_id)  # aligned → same frame

        # ── RGB ──────────────────────────────────────────────────────────────
        bgr = np.asanyarray(color_frame.get_data())
        msg_rgb = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg_rgb.header = header_rgb
        self.pub_rgb.publish(msg_rgb)

        # ── camera_info ──────────────────────────────────────────────────────
        ci = self._camera_info
        ci.header = header_rgb
        self.pub_info.publish(ci)

        # ── Depth (mm, 16UC1) ─────────────────────────────────────────────
        depth_mm = np.asanyarray(depth_frame.get_data())   # uint16, mm
        msg_mm = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
        msg_mm.header = header_depth
        self.pub_depth_mm.publish(msg_mm)

        # ── Depth (metres, 32FC1) ─────────────────────────────────────────
        depth_m = depth_mm.astype(np.float32) * self.depth_scale
        depth_m[depth_m == 0] = np.nan
        msg_m = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        msg_m.header = header_depth
        self.pub_depth_m.publish(msg_m)

    def destroy_node(self):
        self.pipe.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
