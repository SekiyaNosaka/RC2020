# -*- coding: utf-8 -*-
# Server

import numpy as np
import cv2
import os
import time
import sys
import socket
from openvino.inference_engine import IENetwork, IEPlugin

HOST = "*****"  # IP_address
PORT = *****

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()
client, client_address = server.accept()

class PersonAgeGenderRecog():
	def __init__(self):
		self.plugin = IEPlugin(device = "CPU")
		model_age_gender = "~/FP32/age-gender-recognition-retail-0013.xml"

		weights_age_gender = "~/FP32/age-gender-recognition-retail-0013.bin"

		self.net_age_gender = IENetwork(model = model_age_gender, weights = weights_age_gender)
		self.exec_net_age_gender = self.plugin.load(network = self.net_age_gender)

	def age_gender_recog(self):
		self.cap = cv2.VideoCapture(4)
		self.time_start = time.time()

		while (time.time() - self.time_start <= 3.0):
			self.ret, self.frame = self.cap.read()
			self.img = cv2.resize(self.frame, (62, 62))
			self.img = self.img.transpose((2, 0, 1))
			self.img = np.expand_dims(self.img, axis = 0)

			self.out = self.exec_net_age_gender.infer({"data":self.img})
			self.out_age_conv3 = self.out["age_conv3"]
			self.out_prob = self.out["prob"]
			print("age_probability: " + str(self.out_age_conv3))
			print("gender_probability: " + str(self.out_prob) + "\n")
			time.sleep(0.8)

		self.index_max = np.argmax(self.out_prob)

		label_gender = ["female", "male"]

		self.age = int((self.out_age_conv3[0][0][0][0]) * 100)
		self.gender = label_gender[self.index_max]

		print(self.age)
		client.send(str(self.age).encode("utf-8"))
		time.sleep(0.2)
		print(self.gender)
		client.send(str(self.gender).encode("utf-8"))
		self.cap.release()

if __name__ == "__main__":
	recog = PersonAgeGenderRecog()

	cl_rcv1 = client.recv(1024)
	if cl_rcv1 == "Let's personrecog".encode("utf-8"):
		print(cl_rcv1.decode("utf-8"))
		recog.age_gender_recog()
	else:
		sys.exit()
	
	cl_rcv2 = client.recv(1024)
	if cl_rcv2 == "Let's personrecog".encode("utf-8"):
		print(cl_rcv2.decode("utf-8"))
		recog.age_gender_recog()
	else:
		sys.exit()
