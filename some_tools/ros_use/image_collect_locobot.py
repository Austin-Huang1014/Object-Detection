#!/usr/bin/env python
from __builtin__ import True
import numpy as np
import rospy
import math
import torch
import rospkg
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torchvision
import cv2
import os
from torchvision import transforms, utils, datasets
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

os.environ['ROS_IP'] = '10.42.0.1'
bridge = CvBridge()

class JoyCollecter(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.v_temp = 0
        self.o_temp = 0
        self.img = None
        self.v_gain = 0.17
        self.omega_gain = 0.3
        self.joy = None
        rospy.loginfo("[%s] Initializing " % (self.node_name))
        self.pub_car_cmd = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        self.pub_cam_tilt = rospy.Publisher("/tilt/command", Float64, queue_size=1)
        self.sub_joy = rospy.Subscriber("/joy", Joy, self.cbJoy, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", self.Im, , queue_size=1)
    
    def cbJoy(self, joy_msg):
        self.joy = joy_msg
        self.v_temp = self.joy.axes[1]
        self.o_temp = self.joy.axes[3]

    
    def publishControl(self):
        car_cmd_msg = Twist()
        # Left stick V-axis. Up is positive
        car_cmd_msg.linear.x = self.v_temp * self.v_gain
        # Holonomic Kinematics for Normal Driving
        car_cmd_msg.angular.z = self.o_temp * self.omega_gain 
        cv2.imwrite('PATH/'+str(car_cmd_msg.angular.z)+'.jpg', self.img)
        self.pub_car_cmd.publish(car_cmd_msg)
    
    def Im(self, data):
        self.img = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
        self.pub_cam_tilt.publish(0.9)
        self.publishControl()




if __name__ == "__main__":
    rospy.init_node("joy_collecter", anonymous=False)
    joy_collecter = JoyCollecter()
    rospy.spin()
