#! /usr/bin/env python

import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray

class Collect(object):
    def __init__(self):
        self.temp = Int16MultiArray()
        self.temp.data = [0,0,0,0,0,0]
        self.i = 0
        self.a = 0
        self.rgb_sub = rospy.Subscriber('/rgb', Int16MultiArray, self.save, queue_size = 1)
        self.img_sub = rospy.Subscriber('/img', Image, self.Find_angle_dis, queue_size = 10)
    def save(self, data):
        self.temp = data
    def Find_angle_dis(self,data):
        Img = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
        l = Img

       #----------detect blue line------------#
        low_red = np.array([self.temp.data[3],self.temp.data[4],self.temp.data[5]])
        up_red = np.array([self.temp.data[0], self.temp.data[1], self.temp.data[2]])
        Img = cv2.inRange(Img, low_red, up_red)
        Img = 255 -Img
    
    #----------optimi----------------#
        kernel = np.ones((3,3), np.uint8)
        Img = cv2.erode(Img, kernel)
        Img = cv2.dilate(Img,kernel)

    #----------find_contour-------------#
        _, contours, hierarchy = cv2.findContours(Img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,reverse=True)


    #----------place box----------------------------#
        blackbox = cv2.minAreaRect(contours[0])
        (x_min, y_min), (w_min, h_min), angle = blackbox #x,y:center local w,h:weight height
        if abs((w_min)-(h_min)) < 100:
           blackbox = cv2.minAreaRect(contours[1])
           (x_min, y_min), (w_min, h_min), angle = blackbox
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        (x1,y1), (x2,y2), (x3,y3), (x4,y4) = box
        cv2.drawContours(Img,[box],0,(125, 0, 0),3)
        d = x1 - x2
        if w_min > h_min:
             if abs(d) < 10:
                 self.a = 0
                 s = 'S'
             else:
                 self.a = 90 - abs(angle)
                 s = 'L'
        if w_min < h_min:
             if abs(d) < 10:
                 self.a = 0
                 s = 'S'
             else:
                 self.a = 90 - abs(angle)
                 s = 'R'
        print('\n'+s+'_'+str(x_min)+'_'+str(angle))

    #----------show--------#
        cv2.imshow('img',Img)
        cv2.imshow('i',l)
        self.i += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            self.i += 1






if __name__ == '__main__':
    rospy.init_node("image_ana")
    collecter = Collect()
    rospy.spin()
