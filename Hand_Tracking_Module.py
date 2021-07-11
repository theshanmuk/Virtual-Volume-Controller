import cv2
import mediapipe as mp


class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # Convert the BGR image to RGB before processing
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        # print(results.multi_handedness)
        # Draw the hand annotations on the image
        if self.results.multi_hand_landmarks:
            for hand_dots in self.results.multi_hand_landmarks:
                if draw:
                    imgRGB.flags.writeable = True
                    self.mp_drawing.draw_landmarks(
                        img, hand_dots, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.landmark_list = []
        # Hand Box
        xList = []
        yList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, land_mark in enumerate(myHand.landmark):
                h, w, c = img.shape
                x, y = int(land_mark.x * w), int(land_mark.y * h)
                # print(id, x, y)
                xList.append(x)
                yList.append(y)
                self.landmark_list.append([id, x, y])
                if draw:
                    cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)
            # Hand Box
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
        return self.landmark_list, bbox

    def fingersUp(self):

        fingers = []
        # Thumb
        if self.landmark_list[self.tipIds[0]][1] < self.landmark_list[self.tipIds[0] - 1][1]:

            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.landmark_list[self.tipIds[id]][2] < self.landmark_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        length = math.hypot(x2 - x1, y2 - y1)
        m1, m2 = (x1 + x2) // 2, (y1 + y2) // 2
        lineInfo = [x1, y1, x2, y2, m1, m2]
        # Draw the hand annotations on the image
        if draw:
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 1)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED, 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 1)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED, 2)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(img, (m1, m2), 5, (0, 0, 255), cv2.FILLED)
        return length, img, lineInfo
