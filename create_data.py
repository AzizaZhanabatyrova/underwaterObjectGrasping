#!/usr/bin/env python

from geometry_msgs.msg import TwistStamped
import termios, fcntl, sys, os
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
import math

class imageGrabber:

	##Init function, create subscriber and required vars.
	def __init__(self):
		image_sub = rospy.Subscriber("/g500/camera",Image,self.image_callback)
		self.bridge = CvBridge()

	##Image received -> process the image
	def image_callback(self,data):
		global cv_image
		global table
		global start
		global row_to_write
		global my_object

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		cv2.imshow("Image window", cv_image)
		cv2.waitKey(3)

		if (start > 0):

			# Binarize image to segment pipe pixels from the rest
			hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) # change color space

			lower_green = np.array([50, 100, 0])
			upper_green = np.array([70, 255, 255])
			lower_white = np.array([0, 0, 0])
			upper_white = np.array([0, 0, 255])

			green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
			white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
			binarized_image = green_mask + white_mask 


			#binarized_image = cv2.inRange(hsv_image, lower_green, upper_green)
			cv2.imshow("Image window 2", binarized_image)
			# detecting borders in the binary image
			pipe_edges = cv2.Canny(binarized_image,50,150,apertureSize = 3)
			cv2.imshow("Image window 3", pipe_edges)

			# First math descriptor "Compactness"
			num_white_pixels = cv2.countNonZero(binarized_image)
			white_pixels_of_edges = cv2.countNonZero(pipe_edges)
			white_inside_edges = num_white_pixels - white_pixels_of_edges
			math_descriptor1 = white_pixels_of_edges * white_pixels_of_edges / white_inside_edges
			print("\nmath_descriptor1")
			print(math_descriptor1)

			# Second math descriptor "Elongatedness"
			# find centroid of the object
			x_coordinate = 0
			y_coordinate = 0
			x_sum_white2 = 0

			y_sum_white2 = 0
			x_sum_white = 0
			y_sum_white = 0
			xy_sum_white = 0
			while x_coordinate <320:
				while y_coordinate < 240:
					if (binarized_image[y_coordinate, x_coordinate]):
						x_sum_white = x_sum_white + x_coordinate
						y_sum_white = y_sum_white + y_coordinate
						x_sum_white2 = x_sum_white2 + x_coordinate * x_coordinate
						y_sum_white2 = y_sum_white2 + y_coordinate * y_coordinate
						xy_sum_white = xy_sum_white + x_coordinate * y_coordinate	
					y_coordinate = y_coordinate + 1
				x_coordinate = x_coordinate + 1
				y_coordinate = 0
			x_centroid = (x_sum_white / num_white_pixels)
			y_centroid = (y_sum_white / num_white_pixels)
			cv2.circle(cv_image, (x_centroid, y_centroid), 5, (0,255,0), -1)
			cv2.imshow("Image window 4", cv_image)

			u20 = x_sum_white2 - x_centroid * x_sum_white
			u02 = y_sum_white2 - y_centroid * y_sum_white
			u11 = xy_sum_white - y_centroid * x_sum_white
			Imin = (u20 + u02 - math.sqrt(4*u11*u11 + math.pow((u20 - u02),2))) / 2
			Imax = (u20 + u02 + math.sqrt(4*u11*u11 + math.pow((u20 - u02),2))) / 2
			math_descriptor2 = Imin/Imax
			print("\nmath_descriptor2")
			print(math_descriptor2)

			# Third math descriptor "Spreadness"
			math_descriptor3 = (Imin + Imax)/(num_white_pixels*num_white_pixels)
			print("\nmath_descriptor3")
			print(math_descriptor3)

			# Fourth descriptor "Color"
			if (np.sum(green_mask) > 0):
				math_descriptor4 = 1
			else:	
				math_descriptor4 = 0
			print("\nmath_descriptor4")
			print(math_descriptor4)

			# Fifth descriptor "Centroid location"
			if binarized_image[y_centroid,x_centroid] > 0:
				math_descriptor5 = 1
			else:	
				math_descriptor5 = 0				
			print("\nmath_descriptor5")
			print(math_descriptor5)


			# Sixth math descriptor "Hu one"
			eta20 = u20*1000/ (num_white_pixels * num_white_pixels)
			eta02 = u02*1000 / (num_white_pixels * num_white_pixels)
			math_descriptor6 = eta20 + eta02
			print("\nmath_descriptor6")
			print(math_descriptor6)

			# Seventh math descriptor "Hu two"
			eta11 = u11*1000/ (num_white_pixels * num_white_pixels)
			math_descriptor7 = ((eta20 + eta02) * (eta20 + eta02)) + 4*eta11*eta11
			print("\nmath_descriptor7")
			print(math_descriptor7)

			# save data into file
			row = [math_descriptor1, math_descriptor2, math_descriptor3, math_descriptor4, math_descriptor5, math_descriptor6, math_descriptor7, my_object]
			row_to_write = ';'.join(map(str, row)) + '\n'
			f.write(row_to_write)			
			rospy.sleep(0.001)


if __name__ == '__main__':
	global cv_image
	table = []
	start = 0
	my_object = 'hammer'

	#topic to command
	twist_topic="/g500/velocityCommand"
	#base velocity for the teleoperation (0.5 m/s) / (0.5rad/s)
	baseVelocity=0.2

	#Console input variables to teleop it from the console
	fd = sys.stdin.fileno()
	oldterm = termios.tcgetattr(fd)
	newattr = termios.tcgetattr(fd)
	newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
	termios.tcsetattr(fd, termios.TCSANOW, newattr)
	oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
	fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

	##create the publisher
	pub = rospy.Publisher(twist_topic, TwistStamped,queue_size=1)
	rospy.init_node('keyboardCommand2')

	#Create the imageGrabber
	IG=imageGrabber()

	print('\nMENU \nw, s, a, d -> move forward, backward, left, right. \nArrows Up, Down, Left, Right -> Move up, down, rotate left, right. \no -> open the file. \ny -> start image processing. \nu -> pause processing. \nm -> see menu. \nf -> close the data file. \np -> pan. \nh -> hammer. \nb -> bear. \nj -> straight_pipe. \nk -> corner_pipe.\n')

	try:
	    while not rospy.is_shutdown():
		msg = TwistStamped()

		try:
		    c = sys.stdin.read(1)
		    ##Depending on the character set the proper speeds
		    if c=='\n':
			start()
		  	print "Benchmarking Started!"
		    elif c==' ':
			stop()
			print "Benchmark finished!"
		    elif c=='w':
			msg.twist.linear.x=baseVelocity
		    elif c=='s':
			msg.twist.linear.x=-baseVelocity
		    elif c=='a':
			msg.twist.linear.y=-baseVelocity
		    elif c=='d':
			msg.twist.linear.y=baseVelocity
		  
		    elif c=='\x1b':  ##This means we are pressing an arrow!
			c2= sys.stdin.read(1)
			c2= sys.stdin.read(1)
			if c2=='A':
			    msg.twist.linear.z=-baseVelocity
			elif c2=='B':
			    msg.twist.linear.z=baseVelocity
			elif c2=='C':
				msg.twist.angular.z=baseVelocity
			elif c2=='D':
				msg.twist.angular.z=-baseVelocity
		    else:
			print 'Key pressed is ' + c

		    if c=='o':
			f = open('/home/uwsim/uwsim_ws/src/pipefollowing/src/perception_telerobotics/data3.csv', 'a')

		    if c=='y':
			start = 1

		    if c=='u':
			start = 0
		
		    if c=='h':
			my_object = 'hammer'

		    if c=='p':
			my_object = 'pan'

		    if c=='b':
			my_object = 'bear'

		    if c=='j':
			my_object = 'straight_pipe'

		    if c=='k':
			my_object = 'corner_pipe'

		    if c=='f':
			f.close() 

		    if c=='m':
			 print('\nMENU \nw, s, a, d -> move forward, backward, left, right. \nArrows Up, Down, Left, Right -> Move up, down, rotate left, right. \no -> open the file. \ny -> start image processing. \nu -> pause processing. \nm -> see menu. \nf -> close the data file. \np -> pan. \nh -> hammer. \nb -> bear. \nj -> straight_pipe. \nk -> corner_pipe.\n')

		    while c!='':
			c = sys.stdin.read(1)
		except IOError: pass

		##publish the message
		pub.publish(msg)
		rospy.sleep(0.1)

	##Other input stuff
	finally:
	    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
	    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
