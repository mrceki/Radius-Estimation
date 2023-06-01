#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyrealsense2 import pyrealsense2 as rs
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import os
# Modeli yükleme
import joblib

class Estimator:
    def __init__(self, object_ids):
        # Load the pre-trained model
        self.model = joblib.load("/home/user/regression_model.joblib")
        
        # Create a CvBridge object for image conversion
        self.bridge = CvBridge()
        
        # Set the depth scale for converting depth values
        self.depth_scale = 0.001
        
        # Initialize the ROS node
        rospy.init_node('regression_node')
        
        # Subscribe to the object detection topic
        rospy.Subscriber('/yolov7/yolov7', Detection2DArray, self.image_callback)
        
        # Subscribe to the depth image topic
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        
        # Create a publisher for the estimated radius
        self.pub = rospy.Publisher("/radius_estimator", Float32MultiArray, queue_size=10)
        
        # Object IDs to consider for prediction
        self.object_ids = object_ids

    def image_callback(self, detection_array):
        # Bbox to numpy
        predictions = []
        
        # Iterate over the detections in the Detection2DArray message
        for detection in detection_array.detections:
            for result in detection.results:
                # Check if the object ID is in the specified IDs
                if result.id in self.object_ids:
                    bbox = detection.bbox
                    
                    # Get the depth value at the center of the bounding box and convert it using the depth scale
                    depth = (self.depth_image[int(bbox.center.y), int(bbox.center.x)]) * self.depth_scale
                    
                    # Create a feature vector using the depth, width, and height of the bounding box
                    features = np.array([depth, bbox.size_x, bbox.size_y])
                    features = features.reshape(1, -1)
                    
                    # Use the pre-trained model to predict the radius of the object
                    predict = self.model.predict(features)
                    
                    # Add the prediction to the list of predictions
                    predictions.append(predict)
                    print("Radius of object predicted as %f cm" % predict)
        
        # Create a Float32MultiArray message for publishing the predictions
        msg = Float32MultiArray()
        msg.data = np.array(predictions).flatten().tolist()
        
        # Publish the predictions
        self.pub.publish(msg)

    def depth_callback(self, depth_image):
        # Convert the depth image from ROS format to OpenCV format
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")

if __name__ == '__main__':
    # Specify the object IDs to consider for prediction
    object_ids = [47, 49, 80]
    
    # Create an instance of the Estimator class
    estimator = Estimator(object_ids)
    
    # Start the ROS node and spin the main loop
    rospy.spin()
