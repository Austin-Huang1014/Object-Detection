#!/usr/bin/env python

import numpy as np
import rospy
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
import cv2
import time




if __name__ == '__main__':
    rospy.init_node("img_pub", anonymous=True)
    img = cv2.imread('/home/austin/trailnet-testing-Pytorch/duckiefloat_line_follow/src/data/55.jpg')
    img_pub = rospy.Publisher('/img', Image, queue_size = 10)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        img_pub.publish(bridge.cv2_to_imgmsg(img, encoding="passthrough"))