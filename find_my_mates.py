#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Client

import rospy
import numpy as np
import cv2
import tf
import actionlib
import time
import speech_recognition as sr
import socket
from math import pi
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String

HOST = "******"  # IP_address
PORT = *****

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

class FindMyMates():
    def __init__(self): 
        rospy.on_shutdown(self.shutdown)
        self.vel = Twist()
        self.pub_move = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size = 10)
        self.pub_speak = rospy.Publisher("/tts", String, queue_size = 10)
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()
        self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        while not self.ac.wait_for_server(rospy.Duration(5)):
            rospy.loginfo("Waiting for the move_base action server to come up")
        rospy.loginfo("The server comes up")

        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'          # map coordinate system
        self.goal.target_pose.header.stamp = rospy.Time.now()
        self.vel.linear.x = self.vel.linear.y = self.vel.linear.z = 0.0
        self.vel.angular.x = self.vel.angular.y = self.vel.angular.z = 0.0

    def start_speachrecog(self):
        print("-" * 7)
        self.pub_speak.publish("Hi, Please say find")
        time.sleep(2.5)
        while True:
            print("Please say Start sign!")
            with self.mic as source:
                self.r.adjust_for_ambient_noise(source)
                self.audio_start = self.r.listen(source)
            print("Please wait as I process Start sign...")
            try:
                print(self.r.recognize_google(self.audio_start, language = "en-US"))
                if self.r.recognize_google(self.audio_start, language = "en-US") == "find":
                    print("I got it. Find your mate!")
                    self.pub_speak.publish("I got it. Find your mate.")
                    time.sleep(3.5)
                    break
                else:
                    self.pub_speak.publish("Say that once more Please.")
                    time.sleep(3.0)
            except sr.UnknownValueError:
                print("could not understand audio...")
                self.pub_speak.publish("Say that once more Please.")
                time.sleep(3.0)
            except sr.RequestError:
                print("could not request results from Google Speech Recognition Service...")
                self.pub_speak.publish("Say that once more Please.")
                time.sleep(3.0)

    def name_speechrecog(self):
        print("-" * 7)
        self.pub_speak.publish("Please tell me your name.")
        time.sleep(2.5)
        while True:
            print("Please say name!")
            with self.mic as source:
                self.r.adjust_for_ambient_noise(source)
                self.audio_name = self.r.listen(source)
            print("Please wait as I process name...")
            try:
                self.name_recog_str = self.r.recognize_google(self.audio_name, language = "en-US")
                print(self.name_recog_str)
                self.pub_speak.publish("I understand your name.")
                time.sleep(3.5)
                break
            except sr.UnknownValueError:
                self.pub_speak.publish("I couldn't hear it.")
                time.sleep(2.5)
                self.pub_speak.publish("Speak your name again slowly and loudly, please.")
                time.sleep(3.5)

    def person_face_recog(self):
        self.human_face_detection_count = 0
        self.cap = cv2.VideoCapture(4)
        self.casc = cv2.CascadeClassifier("~/haarcascade_frontalface_alt.xml")
        self.pub_speak.publish("I'm going to detect person now.")
        time.sleep(2.5)

        while True:
            self.pub_move.publish(self.vel)
            time.sleep(0.05)
            self.ret, self.frame = self.cap.read()
            self.facerect = self.casc.detectMultiScale(self.frame, scaleFactor = 1.2, minNeighbors = 2, minSize = (1, 1))
            if type(self.facerect) == np.ndarray:
                self.human_face_detection_count += 1
                #print(self.frame)
                #print("facerect: " + str(self.facerect))  # 戻り値 左から 左上x座標, 左上y座標, 横幅, 縦幅
                print("human_face_detection_count: " + str(self.human_face_detection_count) + "\n")

            for rect in self.facerect:
                cv2.rectangle(self.frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (255, 0, 0), thickness = 1)
                self.text = "Person_face"
                self.font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(self.frame, self.text, (rect[0], rect[1] - 10), self.font, 2, (255, 255, 255), 1, cv2.LINE_AA)
                
            cv2.imshow("FACE Image", self.frame)

            self.k = cv2.waitKey(1)
            if self.k == ord("q") or self.human_face_detection_count == 30:
                self.human_face_detection_count = 0
                break

    def wp_navigation(self):
        way_point = [[4.9, -3.8, -1.0 * pi], [0.0, 0.0, -1.0 * pi], [4.0, -3.7, -1.0 * pi], [0.0, 0.0, -1.0 * pi], [999, 999, 999]]
        i = 0

        while not rospy.is_shutdown():
            self.goal.target_pose.pose.position.x =  way_point[i][0]
            self.goal.target_pose.pose.position.y =  way_point[i][1]

            if way_point[i][0] == 999:
                print("Finished because we reached the final destination 999...")
                break
            
            q = tf.transformations.quaternion_from_euler(0, 0, way_point[i][2])
            self.goal.target_pose.pose.orientation = Quaternion(q[0],q[1],q[2],q[3])
            rospy.loginfo("Sending goal: No" + str(i+1))
            self.ac.send_goal(self.goal)
            succeeded = self.ac.wait_for_result(rospy.Duration(90))
            state = self.ac.get_state()

            if succeeded:
                rospy.loginfo("Succeeded: No."+str(i+1)+"("+str(state)+")")
                if i % 2 == 0:
                    if i == 0:
                        self.vel.angular.z = -0.35
                    elif i == 2:
                        self.vel.angular.z = 0.35

                    self.person_face_recog()
                    self.cap.release()
                    cv2.destroyAllWindows()

                    self.pub_speak.publish("Human detected.")
                    time.sleep(2.5)
                    client.send("Let's personrecog".encode("utf-8"))
                    print("sent the 'Let's personrecog' flag.")

                    age_bytes = client.recv(1024)
                    age_str = age_bytes.decode("utf-8")
                    #print(age_str)
                    gender_bytes = client.recv(1024)
                    gender_str = gender_bytes.decode("utf-8")
                    #print(gender_str)

                    self.name_speechrecog()

                if i % 2 == 1:
                    self.pub_speak.publish("About a Person.")
                    time.sleep(2.0)
                    self.pub_speak.publish("The Person's name is " + self.name_recog_str)
                    time.sleep(3.5)
                    self.pub_speak.publish(self.name_recog_str + "'s gender is " + gender_str)
                    time.sleep(3.5)
                    self.pub_speak.publish(self.name_recog_str + "'s age is estimated to be " + age_str)
                    time.sleep(3.5)

            else:
                rospy.loginfo("Failed: No."+str(i+1)+"("+str(state)+")")

            i += 1
      
    def shutdown(self):
        rospy.loginfo("The robot was terminated")
        self.ac.cancel_goal()

if __name__ == '__main__':
    try:
        rospy.init_node('wp_navi')
        find_my_mates = FindMyMates()
        time.sleep(5.0)
        find_my_mates.start_speachrecog()
        find_my_mates.wp_navigation()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Wp_navigation finished.")
