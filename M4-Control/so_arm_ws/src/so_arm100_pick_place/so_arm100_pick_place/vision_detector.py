#!/usr/bin/env python3
"""
vision_detector.py
──────────────────
Classical CV object detection for pick-and-place using RealSense D435.

Objects detected via HSV + shape filtering:
  • banana       — yellow, elongated
  • green_apple  — green, roughly circular
  • red_apple    — red, roughly circular
  • orange       — orange, roughly circular
  • cereal_box   — any dominant hue, large rectangular blob

For each detected object publishes:
  /vision/markers          visualization_msgs/MarkerArray   (RViz spheres + labels)
  /vision/best_object      geometry_msgs/PoseStamped        (highest-confidence pick target)
  /vision/debug_image      sensor_msgs/Image                (annotated BGR)
  /vision/detections       std_msgs/String                  (JSON list of detections)
"""

import json
import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA
from builtin_interfaces.msg import Duration as RosDuration
from cv_bridge import CvBridge


# ── Object profiles ───────────────────────────────────────────────────────────
# Each profile: list of (lower_hsv, upper_hsv) masks  +  shape constraints
# HSV: H 0-180, S 0-255, V 0-255 in OpenCV

OBJECTS = {
    "banana": {
        "color": (0.9, 0.8, 0.1, 1.0),   # RGBA for RViz marker
        "masks": [
            (np.array([18,  80, 100]), np.array([35, 255, 255])),  # yellow
        ],
        "min_area": 1500,
        "aspect_ratio": (1.5, 6.0),   # elongated
        "circularity": (0.0, 0.6),
    },
    "green_apple": {
        "color": (0.1, 0.8, 0.1, 1.0),
        "masks": [
            (np.array([35, 60, 60]),  np.array([80, 255, 255])),   # green
        ],
        "min_area": 2000,
        "aspect_ratio": (0.7, 1.4),
        "circularity": (0.55, 1.0),
    },
    "red_apple": {
        "color": (0.9, 0.1, 0.1, 1.0),
        "masks": [
            (np.array([0,   100, 80]),  np.array([8,  255, 255])),
            (np.array([170, 100, 80]),  np.array([180,255, 255])),
        ],
        "min_area": 2000,
        "aspect_ratio": (0.7, 1.4),
        "circularity": (0.55, 1.0),
    },
    "orange": {
        "color": (1.0, 0.5, 0.0, 1.0),
        "masks": [
            (np.array([8,  150, 100]), np.array([18, 255, 255])),  # orange
        ],
        "min_area": 2000,
        "aspect_ratio": (0.7, 1.4),
        "circularity": (0.55, 1.0),
    },
    "cereal_box": {
        "color": (0.6, 0.3, 0.0, 1.0),
        "masks": [
            (np.array([0,   0,  180]), np.array([180, 50, 255])),  # near-white/grey
            (np.array([10,  80, 100]), np.array([40, 255, 220])),  # warm tones
        ],
        "min_area": 5000,
        "aspect_ratio": (0.4, 3.0),
        "circularity": (0.0, 0.6),
    },
}

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ── Node ─────────────────────────────────────────────────────────────────────

class VisionDetectorNode(Node):

    def __init__(self):
        super().__init__("vision_detector")

        self.declare_parameter("min_depth_m",   0.10)
        self.declare_parameter("max_depth_m",   1.50)
        self.declare_parameter("camera_frame",  "camera_color_optical_frame")
        self.declare_parameter("world_frame",   "world")
        self.declare_parameter("patch_size",    7)       # depth median patch half-size

        self.min_depth   = self.get_parameter("min_depth_m").value
        self.max_depth   = self.get_parameter("max_depth_m").value
        self.cam_frame   = self.get_parameter("camera_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.patch       = self.get_parameter("patch_size").value

        self.bridge       = CvBridge()
        self.camera_info  = None
        self.depth_image  = None

        be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=2
        )

        # Subscribers
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self._info_cb,  be)
        self.create_subscription(Image,      "/camera/aligned_depth/image_raw", self._depth_cb, be)
        self.create_subscription(Image,      "/camera/color/image_raw",  self._rgb_cb,   be)

        # Publishers
        self.pub_markers = self.create_publisher(MarkerArray,   "/vision/markers",       10)
        self.pub_best    = self.create_publisher(PoseStamped,   "/vision/best_object",   10)
        self.pub_debug   = self.create_publisher(Image,         "/vision/debug_image",   10)
        self.pub_json    = self.create_publisher(String,        "/vision/detections",    10)

        self.get_logger().info("VisionDetectorNode ready — waiting for camera topics…")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def _depth_cb(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _rgb_cb(self, msg: Image):
        if self.camera_info is None or self.depth_image is None:
            return

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self._detect_all(bgr, msg.header)

        self._publish_markers(detections, msg.header)
        self._publish_best(detections, msg.header)
        self._publish_json(detections)

        debug_img = self._draw_debug(bgr.copy(), detections)
        dbg_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        dbg_msg.header = msg.header
        self.pub_debug.publish(dbg_msg)

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_all(self, bgr: np.ndarray, header) -> list[dict]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        results = []

        for obj_name, profile in OBJECTS.items():
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            for lo, hi in profile["masks"]:
                mask |= cv2.inRange(hsv, lo, hi)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < profile["min_area"]:
                    continue

                # Shape filters
                rect   = cv2.minAreaRect(cnt)
                w, h   = sorted([rect[1][0], rect[1][1]])
                if h < 1:
                    continue
                ar = h / w
                ar_lo, ar_hi = profile["aspect_ratio"]
                if not (ar_lo <= ar <= ar_hi):
                    continue

                perim = cv2.arcLength(cnt, True)
                circ  = 4 * math.pi * area / (perim ** 2) if perim > 0 else 0
                ci_lo, ci_hi = profile["circularity"]
                if not (ci_lo <= circ <= ci_hi):
                    continue

                # Centroid
                M  = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 3D position from depth
                pt3d = self._backproject(cx, cy)
                if pt3d is None:
                    continue

                # Confidence = normalized area * circularity score
                confidence = min(1.0, area / 20000) * (0.5 + 0.5 * circ)

                results.append({
                    "name":       obj_name,
                    "cx":         cx,
                    "cy":         cy,
                    "area":       area,
                    "contour":    cnt,
                    "point3d":    pt3d,      # (x,y,z) in camera frame (metres)
                    "confidence": round(confidence, 3),
                    "color":      profile["color"],
                })

        # Sort by confidence descending
        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results

    def _backproject(self, cx: int, cy: int):
        """Return (x,y,z) metres in camera frame, or None if invalid depth."""
        h, w = self.depth_image.shape
        p = self.patch
        x0, x1 = max(0, cx - p), min(w, cx + p + 1)
        y0, y1 = max(0, cy - p), min(h, cy + p + 1)
        patch = self.depth_image[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)]
        if valid.size < 5:
            return None

        z = float(np.median(valid))
        fx  = self.camera_info.k[0]
        fy  = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]
        x = (cx - ppx) * z / fx
        y = (cy - ppy) * z / fy
        return (x, y, z)

    # ── Publishers ─────────────────────────────────────────────────────────────

    def _publish_markers(self, detections: list[dict], header):
        array = MarkerArray()

        # Delete all previous markers first
        del_marker = Marker()
        del_marker.header = header
        del_marker.action = Marker.DELETEALL
        array.markers.append(del_marker)

        for i, det in enumerate(detections):
            x, y, z = det["point3d"]
            r, g, b, a = det["color"]

            # Sphere at object centre
            sphere = Marker()
            sphere.header  = header
            sphere.ns      = "objects"
            sphere.id      = i * 2
            sphere.type    = Marker.SPHERE
            sphere.action  = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.06
            sphere.color   = ColorRGBA(r=r, g=g, b=b, a=a)
            sphere.lifetime = RosDuration(sec=1, nanosec=0)
            array.markers.append(sphere)

            # Text label above the sphere
            label = Marker()
            label.header  = header
            label.ns      = "labels"
            label.id      = i * 2 + 1
            label.type    = Marker.TEXT_VIEW_FACING
            label.action  = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = z + 0.08
            label.pose.orientation.w = 1.0
            label.scale.z = 0.04
            label.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            label.text    = f"{det['name']}\n{det['confidence']:.2f}"
            label.lifetime = RosDuration(sec=1, nanosec=0)
            array.markers.append(label)

            # Arrow pointing down to object (pick approach direction)
            arrow = Marker()
            arrow.header  = header
            arrow.ns      = "approach"
            arrow.id      = i * 2 + 100
            arrow.type    = Marker.ARROW
            arrow.action  = Marker.ADD
            # Arrow from above to object
            start = Point(x=x, y=y - 0.15, z=z)
            end   = Point(x=x, y=y,        z=z)
            arrow.points  = [start, end]
            arrow.scale.x = 0.01   # shaft diameter
            arrow.scale.y = 0.02   # head diameter
            arrow.color   = ColorRGBA(r=r, g=g, b=b, a=0.7)
            arrow.lifetime = RosDuration(sec=1, nanosec=0)
            array.markers.append(arrow)

        self.pub_markers.publish(array)

    def _publish_best(self, detections: list[dict], header):
        if not detections:
            return
        best = detections[0]
        x, y, z = best["point3d"]

        ps = PoseStamped()
        ps.header = header
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0
        self.pub_best.publish(ps)

    def _publish_json(self, detections: list[dict]):
        payload = [
            {
                "name":       d["name"],
                "x":          round(d["point3d"][0], 4),
                "y":          round(d["point3d"][1], 4),
                "z":          round(d["point3d"][2], 4),
                "confidence": d["confidence"],
            }
            for d in detections
        ]
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_json.publish(msg)

    # ── Debug drawing ─────────────────────────────────────────────────────────

    def _draw_debug(self, bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        for det in detections:
            r, g, b, _ = det["color"]
            color_bgr = (int(b * 255), int(g * 255), int(r * 255))

            cv2.drawContours(bgr, [det["contour"]], -1, color_bgr, 2)
            cx, cy = det["cx"], det["cy"]
            cv2.circle(bgr, (cx, cy), 5, color_bgr, -1)

            x3, y3, z3 = det["point3d"]
            label = f"{det['name']} ({z3:.2f}m)"
            cv2.putText(bgr, label, (cx + 8, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
            cv2.putText(bgr, f"conf:{det['confidence']:.2f}", (cx + 8, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)

        # HUD
        n = len(detections)
        cv2.putText(bgr, f"Detections: {n}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if detections:
            best = detections[0]
            cv2.putText(bgr, f"Best: {best['name']}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return bgr


def main(args=None):
    rclpy.init(args=args)
    node = VisionDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
