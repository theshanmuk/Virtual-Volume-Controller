import imp
import cv2
import numpy as np
import math
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mp_drawing = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert the BGR image to RGB before processing
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_handedness)
        # Draw the hand annotations on the image
        if self.results.multi_hand_landmarks:
            for hand_dots in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img, hand_dots, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, land_mark in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(land_mark.x*w), int(land_mark.y*h)
                #print(id, cx, cy)
                self.landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.landmark_list
