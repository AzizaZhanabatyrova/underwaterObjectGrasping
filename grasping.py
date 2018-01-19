#!/usr/bin/env python

# This file is using trained model to classify objects and to perform grasping of the object

from geometry_msgs.msg import TwistStamped
import termios, fcntl, sys, os
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sklearn.neural_network import MLPClassifier
from sensor_msgs.msg import Image, Range
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from numpy.linalg import inv
import math
import pickle
import tf

class imageGrabber:

	##Init function, create subscriber and required vars.
	def __init__(self):
		image_sub = rospy.Subscriber("/g500/camera",Image,self.image_callback)
		depth_image_sub = rospy.Subscriber("/g500/range",Range,self.range_callback)
		odometry = rospy.Subscriber("/uwsim/girona500_odom",Odometry,self.odometry_callback)
		self.bridge = CvBridge()

	# To receive data about distance
	def range_callback(self,data):
		global range_distance
		range_distance = data.range

	# To receive data from odometry
	def odometry_callback(self,data):
		global x_odom
		global y_odom
		global z_odom
		x_odom = data.pose.pose.position.x
		y_odom = data.pose.pose.position.y
		z_odom = data.pose.pose.position.z

	##Image received -> process the image
	def image_callback(self,data):
		global cv_image
		global object_class
		global start
		global x_grasp
		global y_grasp
		global row_to_write
		global centered

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		cv2.imshow("Image window", cv_image)
		cv2.waitKey(3)

		if (start > 0 and centered == 0):

			# Binarize image to segment pipe pixels from the rest
			hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) # change color space

			lower_green = np.array([50, 100, 0])
			upper_green = np.array([70, 255, 255])
			lower_white = np.array([0, 0, 0])
			upper_white = np.array([0, 0, 255])

			green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
			white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
			binarized_image = green_mask + white_mask 
			cv2.imshow("Image window 2", binarized_image)

			# detecting borders in the binary image
			obj_edges = cv2.Canny(binarized_image,50,150,apertureSize = 3)
			cv2.imshow("Image window 3", obj_edges)

			# First math descriptor "Compactness"
			num_white_pixels = cv2.countNonZero(binarized_image)
			if (num_white_pixels < 10):
				print('No object was found')
				return;
			white_pixels_of_edges = cv2.countNonZero(obj_edges)
			white_inside_edges = num_white_pixels - white_pixels_of_edges
			math_descriptor1 = white_pixels_of_edges * white_pixels_of_edges / white_inside_edges

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
			u20 = x_sum_white2 - x_centroid * x_sum_white
			u02 = y_sum_white2 - y_centroid * y_sum_white
			u11 = xy_sum_white - y_centroid * x_sum_white
			Imin = (u20 + u02 - math.sqrt(4*u11*u11 + math.pow((u20 - u02),2))) / 2
			Imax = (u20 + u02 + math.sqrt(4*u11*u11 + math.pow((u20 - u02),2))) / 2
			math_descriptor2 = Imin/Imax

			# Third math descriptor "Spreadness"
			math_descriptor3 = (Imin + Imax)/(num_white_pixels*num_white_pixels)

			# Fourth descriptor "Color"
			if (np.sum(green_mask) > 0):
				math_descriptor4 = 1
			else:	
				math_descriptor4 = 0

			# Fifth descriptor "Centroid location"
			if binarized_image[y_centroid,x_centroid] > 0:
				math_descriptor5 = 1
			else:	
				math_descriptor5 = 0			
	
			# Predict class for a new object
			descriptors = np.array([math_descriptor1, math_descriptor2, math_descriptor3, math_descriptor4, math_descriptor5])
			object_prediction = clf.predict_proba([descriptors])
			object_class = np.argmax(object_prediction)

			if object_class == 0:
				print('bear')
			elif object_class == 1:
				print('pan')
			elif object_class == 2:
				print('hammer')
			elif object_class == 3:
				print('corner_pipe')
			elif object_class == 4:
				print('straight_pipe')
			else:
				print('object not identified')

			rospy.sleep(0.001)
			max_dist = 0
			x_max = 0
			y_max = 0
			x_coord = 0
			y_coord = 0

			if object_class == 0 or object_class == 1: # bear or pan
				max_dist = 0
				x_max = 0
				y_max = 0
				x_coord = 0
				y_coord = 0
				while x_coord <320:
					while y_coord < 240:
						if (obj_edges[y_coord, x_coord]):
							dist = math.sqrt(math.pow(x_coord-x_centroid,2) + math.pow(y_coord-y_centroid,2))
							if (dist > max_dist):
								max_dist = dist
								x_max = x_coord
								y_max = y_coord	
						y_coord = y_coord + 1
					x_coord = x_coord + 1
					y_coord = 0
	
				x_grasp = (x_max - x_centroid)/1.5 + x_centroid
				y_grasp = (y_max - y_centroid)/1.5 + y_centroid

			elif object_class == 2: # Hammer 
				x_grasp = x_centroid
				y_grasp = y_centroid

			elif object_class == 4: # Straight pipe
				pipe_lines = cv2.HoughLinesP(obj_edges,1,np.pi/180,50,60,20)
				if np.all(pipe_lines!=None): 
      					for x1,y1,x2,y2 in pipe_lines[0]: 
        					cv2.line(cv_image,(x1,y1),(x2,y2),(0,255,0),2) 
			        cv2.imshow("Image window 5", cv_image) 
			        cv2.waitKey(3)
				# coordinates of two lines
			        x1, y1, x2, y2 = pipe_lines[0][0]
			        x1_2, y1_2, x2_2, y2_2 = pipe_lines[0][1]

			        # Calculate center of visible pipe
			        x3 = (x1 + x2) / 2
			        y3 = (y1 + y2) / 2  

			        x3_2 = (x1_2 + x2_2) / 2
			        y3_2 = (y1_2 + y2_2) / 2 

			        x4 = (x3 + x3_2) / 2
			        y4 = (y3 + y3_2) / 2

				x_grasp = x4
				y_grasp = y4


			else: # corner pipe
				min_dist = 10000000000
				x_min = 0
				y_min = 0
				x_coord = 0
				y_coord = 0
				# Find first grasping point
				while x_coord <320:
					while y_coord < 240:
						if (obj_edges[y_coord, x_coord]):
							dist = math.sqrt(math.pow(x_coord-x_centroid,2) + math.pow(y_coord-y_centroid,2))
							if (dist < min_dist):
								min_dist = dist
								x_min = x_coord
								y_min = y_coord	
						y_coord = y_coord + 1
					x_coord = x_coord + 1
					y_coord = 0
	
				x_grasp = (x_min - x_centroid)*1.2 + x_centroid
				y_grasp = (y_min - y_centroid)*1.2 + y_centroid
				g1_dist = min_dist
				
				min_dist = 100000000000
				min_angle = 3.14
				x_coord = 0
				y_coord = 0
				# Find second grasping point
				pipe_lines = cv2.HoughLinesP(obj_edges,1,np.pi/180,50,60,20)
				if np.all(pipe_lines!=None): 
      					for x1,y1,x2,y2 in pipe_lines[0]: 
        					cv2.line(cv_image,(x1,y1),(x2,y2),(0,255,0),2) 
			        cv2.imshow("Image window 5", cv_image) 
			        cv2.waitKey(3)
				# coordinates of two lines
			        x1, y1, x2, y2 = pipe_lines[0][0]
			        x1_2, y1_2, x2_2, y2_2 = pipe_lines[0][1]
			        x1_3, y1_3, x2_3, y2_3 = pipe_lines[0][2]
			        x1_4, y1_4, x2_4, y2_4 = pipe_lines[0][3]

			        # Calculate center of visible pipe
			        x3 = (x1 + x2) / 2
			        y3 = (y1 + y2) / 2  

			        x3_2 = (x1_2 + x2_2) / 2
			        y3_2 = (y1_2 + y2_2) / 2 

			        x4 = (x3 + x3_2) / 2 # Canter of pipe
			        y4 = (y3 + y3_2) / 2
				
			        x3_3 = (x1_3 + x2_3) / 2
			        y3_3 = (y1_3 + y2_3) / 2  

			        x3_4 = (x1_4 + x2_4) / 2
			        y3_4 = (y1_4 + y2_4) / 2 

			        x4_2 = (x3_3 + x3_4) / 2 # Canter of pipe
			        y4_2 = (y3_3 + y3_4) / 2

				x_grasp = x4
				y_grasp = y4
				x_grasp2 = x4_2
				y_grasp2 = y4_2

			# display grasping point
			cv2.circle(cv_image, (int(x_grasp), int(y_grasp)), 5, (127,0,255), -1)
			if (object_class == 3):
				cv2.circle(cv_image, (int(x_grasp2), int(y_grasp2)), 5, (127,0,255), -1)
			cv2.imshow("Image window 4", cv_image)
			
			if (object_class != 3 and object_class != 4):
				self.moveToCenter(x_grasp, y_grasp,range_distance)

		if (centered):
			cv2.circle(cv_image, (int(x_grasp), int(y_grasp)), 5, (127,0,255), -1)
			cv2.imshow("Image window 4", cv_image)


	def moveToCenter(self, x_grasp, y_grasp, range_distance):
		global centered
		global grasping_point_in_gripper_list

		tol = 6
		gain = 0.5/100
		
		error_x = (160 - x_grasp)
		error_y = (120 - y_grasp)

		if ((abs(error_x) > tol or abs(error_y) > tol) and centered != 1):
			msg.twist.linear.x = gain*error_y
			msg.twist.linear.y = -gain*error_x

		if (abs(error_x) < tol and abs(error_y) < tol and centered != 1):
			centered = 1
			self.calculate_frame()
			grasping_point_in_gripper  = np.dot(cTg,np.array([0,0,range_distance, 1]))
			grasping_point_in_gripper_list = grasping_point_in_gripper.tolist()
			self.move_robot()
			
		pub.publish(msg)

	# move until the gripper is at the desired position
	def move_robot(self):
		initial = [x_odom, y_odom, z_odom]
		x_final = initial[0] + grasping_point_in_gripper_list[0][0]
		y_final = initial[1] + grasping_point_in_gripper_list[0][1]
		z_final = initial[2] + grasping_point_in_gripper_list[0][2] 
		thres = 0.1
		print("=========================================================")
		print(x_odom)
		print(y_odom)
		print(z_odom)
		print(x_final)
		print(y_final)
		print(z_final)

		while ((abs(x_odom - x_final) > thres) or (abs(y_odom - y_final) > thres) or (abs(z_odom - z_final) > thres)):
			msg.twist.linear.x=0.1 * -(x_odom - x_final)
			msg.twist.linear.y=0.1 * -(y_odom - y_final)
			msg.twist.linear.z=0.1 * -(z_odom - z_final)
			pub.publish(msg)
		return;

	# To calculate transformation matrix from Camera to Gripper
	def calculate_frame(self):
		global cTg
	
		try:
			(trans, rot) = listener.lookupTransform('/girona500/part0', '/bowtech', rospy.Time(0))

			cTp0_rot = tf.transformations.quaternion_matrix(rot)
			cTp0_trans = tf.transformations.translation_matrix(trans)
			cTp0 = np.dot(cTp0_trans,cTp0_rot) # From camera to part0

			kbTp0_rot_qua = tf.transformations.quaternion_from_euler(0, 0, 0, 'rxyz')
			kbTp0_rot = tf.transformations.quaternion_matrix(kbTp0_rot_qua)
			kbTp0_trans = tf.transformations.translation_matrix([0, 0, 0.13])
			kbTp0 = np.dot(kbTp0_trans,kbTp0_rot) # Kinematic base to part 0

			(trans3, rot3) = listener.lookupTransform('/girona500/kinematic_base', '/girona500/part1', rospy.Time(0))

			p1Tkb_rot = tf.transformations.quaternion_matrix(rot3)
			p1Tkb_trans = tf.transformations.translation_matrix(trans3)
			p1Tkb = np.dot(p1Tkb_trans,p1Tkb_rot) # From part1 to kinematic base

			(trans4, rot4) = listener.lookupTransform('/girona500/part1', '/girona500/part2', rospy.Time(0))	

			p2Tp1_rot = tf.transformations.quaternion_matrix(rot4)
			p2Tp1_trans = tf.transformations.translation_matrix(trans4)
			p2Tp1 = np.dot(p2Tp1_trans,p2Tp1_rot) # From part2 to part1

			(trans5, rot5) = listener.lookupTransform('/girona500/part2', '/girona500/part3', rospy.Time(0))

			p3Tp2_rot = tf.transformations.quaternion_matrix(rot5)
			p3Tp2_trans = tf.transformations.translation_matrix(trans5)
			p3Tp2 = np.dot(p3Tp2_trans,p3Tp2_rot) # From part3 to part2

			(trans6, rot6) = listener.lookupTransform('/girona500/part3', '/girona500/part4_base', rospy.Time(0))

			gTp3_rot = tf.transformations.quaternion_matrix(rot6)
			gTp3_trans = tf.transformations.translation_matrix(trans6)
			gTp3 = np.dot(gTp3_trans,gTp3_rot) # From gripper to part3

			gTp2 = np.dot(p3Tp2,gTp3) # From gripper to part 2
			p2Tkb = np.dot(p1Tkb,p2Tp1) # From part 2 to kinematic base
			gTkb = np.dot(p2Tkb,gTp2) # From gripper to kinematic base
			gTp0 = np.dot(kbTp0, gTkb) # From gripper to part0
	
			p0Tg = inv(np.matrix(gTp0)) # From part 0 to gripper
			cTg = np.dot(p0Tg,cTp0) # From camera to gripper
	

		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			return False;

		return True;


if __name__ == '__main__':
	global clf
	global cv_image
	global pub, msg
	global listener
	global centered
	centered = 0
	start = 0
	
	#topic to command
	twist_topic="/g500/velocityCommand"
	#base velocity for the teleoperation (0.5 m/s) / (0.5rad/s)
	baseVelocity=0.6

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

	# Subscriber
	listener = tf.TransformListener()

	#Create the imageGrabber
	IG=imageGrabber()

	# To load the trained model
	with open('/home/uwsim/uwsim_ws/src/pipefollowing/src/perception_telerobotics/trained_model4.pkl', 'rb') as f:
		clf = pickle.load(f)

	# teleoperation
	print('\nMENU \nw, s, a, d -> move forward, backward, left, right. \nArrows Up, Down, Left, Right -> Move up, down, rotate left, right. \ny -> start image processing. \nu -> pause processing. \nm -> see menu. \n')

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

		    if c=='y':
			start = 1

		    if c=='u':
			start = 0

		    if c=='m':
			 print('\nMENU \nw, s, a, d -> move forward, backward, left, right. \nArrows Up, Down, Left, Right -> Move up, down, rotate left, right. \ny -> start image processing. \nu -> pause processing. \nm -> see menu. \n')

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
