#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray
import cv2
import os
import edit_dataset
class DatasetCreator:
    def __init__(self):
        self.bridge = CvBridge()  # Bridge object for image conversion
        self.depth_scale = 0.001  # Scaling factor for depth values
        self.previous_depth = None  # Previous depth value
        self.image_sub = rospy.Subscriber('/yolov7/yolov7', Detection2DArray, self.detection_callback)  # Subscriber for object detection
        self.depth_image_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)  # Subscriber for depth image
        self.dataset_path = "/home/user/dataset/"  # Dataset directory path
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
    
    def detection_callback(self, detection_array):
        radius = 3.4
        for detection in detection_array.detections:
            if detection.results[0].id == 32 or detection.results[0].id == 47 :  # If object ID is 32 or 47
                print("detected")
                bbox = detection.bbox
                # If depth data is available
                if hasattr(self, 'depth_image'):
                    depth = (self.depth_image[int(bbox.center.y), int(bbox.center.x)]) * self.depth_scale
                    print(depth)
                    # Skip image capture if depth difference exceeds threshold
                    if self.previous_depth is not None and abs(depth - self.previous_depth) >= 0.10:
                        print(f"Skipping image capture. Depth difference ({abs(depth - self.previous_depth):.2f}) exceeds threshold.")
                        if abs(depth - self.previous_depth) >= 0.50:
                            return
                        self.previous_depth = depth
                        return
                    # Save the data
                    image = self.bridge.imgmsg_to_cv2(detection.source_img)
                    try:
                        cv2.imwrite(f'{self.dataset_path}/{rospy.get_rostime().to_nsec()}.jpg', image)
                        with open(f'{self.dataset_path}/{rospy.get_rostime().to_nsec()}.txt', 'w') as f:
                            f.write(f'{radius} {depth} {bbox.size_x} {bbox.size_y}')
                    except:
                        print("Can't write output!")
                    self.previous_depth = depth

    def depth_callback(self, depth_image):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")

if __name__ == '__main__':
    rospy.init_node('dataset_creator')
    dataset_creator = DatasetCreator()
    rospy.spin()
    edit_dataset.combine_and_shuffle_csv

