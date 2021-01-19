#!/usr/bin/env python

import numpy as np
import rospy
#import picamera
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
import cv2
import time
import message_filters


class Collect(object):
    def __init__(self):
        self.IMG = None
        self.i = 301
        self.k = 0
        self.I = 0
        self.G = 0
        #image_sub = message_filters.Subscriber('/box_pred/img_roi', Image)
        #info_sub = message_filters.Subscriber('/box_pred/img', Image)
        #info_sub = rospy.Subscriber('/camera/color/image_raw', Image)

        #ts = message_filters.TimeSynchronizer([image_sub,info_sub], 10)
        #ts.registerCallback(self.save)
        self.img = rospy.Subscriber('/camera/color/image_raw', Image, self.save, queue_size = 1)
        #self.img = rospy.Subscriber('/box_pred/img_roi',Image, self.save, queue_size = 1)
        #self.Img = rospy.Subscriber('/box_pred/img', Image, self.Save, queue_size = 1)
    #def save(self, data1,data2):
    def save(self, data1):
        self.IMG1 = bridge.imgmsg_to_cv2(data1, desired_encoding = "bgr8")
        self.IMG = cv2.resize(self.IMG1,(640,480))
        cv2.imwrite("/home/austin/bag/backpack_173/"+'backpack_add_'+str(self.i)+'.jpg', self.IMG)
        #self.IMG2 = bridge.imgmsg_to_cv2(data2, desired_encoding = "bgr8")
        #self.IMG3 = bridge.imgmsg_to_cv2(data3, desired_encoding = "bgr8")
        #if ((self.i)%6 == 0):
            #cv2.imwrite("/home/austin/bag/BAG_img/missile/"+'missile_'+str(self.I)+'.jpg', self.IMG1)
            #cv2.imwrite("/home/austin/bag/BAG_img/origin/"+'ROI_'+str(self.I)+'.jpg', self.IMG2)
            #self.I += 1
            #cv2.imwrite("/home/austin/DataSet/can_2"+str(self.i)+'_can'+'.jpg', self.IMG)
        rospy.loginfo(str(self.i))
        self.i += 1


if __name__ == "__main__":
    rospy.init_node("collect")
    rospy.loginfo('ss')
    collecter = Collect()
    rospy.spin()
