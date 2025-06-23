#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import math

class LineExtractorLive(Node):
    def __init__(self):
        super().__init__('line_extractor_live')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        self.marker_pub = self.create_publisher(Marker, '/visualization_markers', 10)

    def scan_callback(self, msg):
        try:
            angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
            ranges = np.array(msg.ranges)

            valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
            ranges = ranges[valid]
            angles = angles[valid]

            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)
            points = np.column_stack((x, y))

            if len(points) > 30:
                lines = self.extract_lines(points)
                self.visualize_lines(lines)
        except Exception as e:
            self.get_logger().error(f"Error in scan processing: {str(e)}")

    def extract_lines(self, points, dist_threshold=0.02, min_points=18, max_iter=200):
        lines = []
        remaining = points.copy()

        for _ in range(max_iter):
            if len(remaining) < min_points:
                break

            idx = np.random.choice(len(remaining), size=2, replace=False)
            p1, p2 = remaining[idx]

            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0]*p1[1] - p1[0]*p2[1]
            norm = math.hypot(a, b)
            if norm == 0:
                continue
            distances = np.abs(a*remaining[:,0] + b*remaining[:,1] + c) / norm

            inliers = remaining[distances < dist_threshold]
            if len(inliers) < min_points:
                continue

            x = inliers[:, 0]
            y = inliers[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c_fit = np.linalg.lstsq(A, y, rcond=None)[0]

            x_min, x_max = np.min(x), np.max(x)
            p_start = np.array([x_min, m * x_min + c_fit])
            p_end = np.array([x_max, m * x_max + c_fit])

            lines.append(((m, c_fit), (p_start, p_end)))
            remaining = remaining[distances >= dist_threshold]

        return lines

    def visualize_lines(self, lines):
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        self.marker_pub.publish(clear_marker)

        for i, ((m, c), (p1_arr, p2_arr)) in enumerate(lines):
            marker = Marker()
            marker.header.frame_id = "base_scan"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lines"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.04
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            p1 = Point(x=p1_arr[0], y=p1_arr[1], z=0.0)
            p2 = Point(x=p2_arr[0], y=p2_arr[1], z=0.0)
            marker.points = [p1, p2]
            self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = LineExtractorLive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
