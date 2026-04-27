#!/usr/bin/env python3
"""
vision_node.py
──────────────
Subscribes to the simulated ZED2 RGB and depth topics.
Detects the red cube in the scene using HSV thresholding,
then uses the depth image + camera intrinsics to compute
the 3-D position of the object in the camera frame and
broadcasts it as a TF frame ("detected_object") and a
geometry_msgs/PointStamped on /vision/object_position.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.bridge = CvBridge()
        self.camera_info: CameraInfo | None = None
        self.depth_image: np.ndarray | None = None

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publisher
        self.obj_pub = self.create_publisher(PointStamped, "/vision/object_position", 10)
        self.debug_pub = self.create_publisher(Image, "/vision/debug_image", 10)

        # Subscribers
        self.create_subscription(CameraInfo, "/zed2/rgb/camera_info",
                                 self._camera_info_cb, qos)
        self.create_subscription(Image, "/zed2/depth/depth_registered",
                                 self._depth_cb, qos)
        self.create_subscription(Image, "/zed2/rgb/image_rect_color",
                                 self._rgb_cb, qos)

        self.get_logger().info("VisionNode started – waiting for ZED2 topics…")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def _depth_cb(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _rgb_cb(self, msg: Image):
        if self.camera_info is None or self.depth_image is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        point_3d = self._detect_red_cube(rgb, msg.header)

        if point_3d is not None:
            self._publish_tf(point_3d, msg.header)
            self._publish_point(point_3d, msg.header)

        # Always publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect_red_cube(self, bgr: np.ndarray, header) -> np.ndarray | None:
        """Return (x, y, z) in camera frame of the largest red blob, or None."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Red wraps around 0/180 in OpenCV HSV
        mask1 = cv2.inRange(hsv, np.array([0,  120, 70]), np.array([10,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160,120, 70]), np.array([180, 255, 255]))
        mask  = cv2.bitwise_or(mask1, mask2)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:   # ignore tiny blobs
            return None

        # Centroid in pixel space
        M  = cv2.moments(largest)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Draw on debug image
        cv2.drawContours(bgr, [largest], -1, (0, 255, 0), 2)
        cv2.circle(bgr, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(bgr, f"cube ({cx},{cy})", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Depth at centroid (median of 5×5 patch for robustness)
        h, w = self.depth_image.shape
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        patch = self.depth_image[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None

        depth = float(np.median(valid))   # metres

        # Back-project to 3-D using pinhole model
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]

        x3d = (cx - ppx) * depth / fx
        y3d = (cy - ppy) * depth / fy
        z3d = depth

        self.get_logger().info(
            f"Detected cube → camera frame: x={x3d:.3f} y={y3d:.3f} z={z3d:.3f}",
            throttle_duration_sec=1.0,
        )
        return np.array([x3d, y3d, z3d])

    # ── publish ───────────────────────────────────────────────────────────────

    def _publish_tf(self, point: np.ndarray, header):
        t = TransformStamped()
        t.header = header
        t.child_frame_id = "detected_object"
        t.transform.translation.x = float(point[0])
        t.transform.translation.y = float(point[1])
        t.transform.translation.z = float(point[2])
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def _publish_point(self, point: np.ndarray, header):
        msg = PointStamped()
        msg.header = header
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
        self.obj_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
