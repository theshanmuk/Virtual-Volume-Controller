import cv2
import numpy as np
import time
import Hand_Tracking_Module as htm
# Volume_Packages
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
pTime = 0
detector = htm.handDetector(detectionCon=0.7)

# BASIC_VOLUME_TOOL
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_bar = 320
vol_per = 0

while True:
    # Find hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    landmark_list, _ = detector.findPosition(img, draw=False)
    if len(landmark_list) != 0:
        #print(landmark_list[4], landmark_list[8])
        x1, y1 = landmark_list[4][1], landmark_list[4][2]  # Index_Finger
        x2, y2 = landmark_list[8][1], landmark_list[8][2]  # Thumb_Finger
        length = math.hypot(x2-x1, y2-y1)  # Length b/w Fingers
        m1, m2 = (x1+x2) // 2, (y1+y2) // 2  # Mid Point
        # print(int(length))
        # Check Thumb & Index fingers are open
        fingers = detector.fingersUp()
        # print(fingers)
        if fingers[0] == 0 or fingers[0] == 1 and fingers[2] == 1:
            pass

        else:
            print(fingers)
            # Draw the hand annotations on the image
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 1)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED, 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 1)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED, 2)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(img, (m1, m2), 5, (0, 0, 255), cv2.FILLED)

            # Fingers Length Range = 20 to 150 (accuracy) changing to required values
            # Volume Range = -65 to 0 from pycaw
            # -----------------Interpolation--------------------
            vol = np.interp(length, [20, 100], [-65, 0])
            vol_bar = np.interp(length, [20, 100], [320, 100])
            vol_per = np.interp(length, [20, 100], [0, 100])
            # changing vol from -65 to 0
            volume.SetMasterVolumeLevel(vol, None)
            # print(int(length), vol, vol_bar, vol_per)

            # Draw the volume annotations on the image
        if length < 20:
            cv2.circle(img, (m1, m2), 5, (255, 255, 255), cv2.FILLED)

    # Volume BAR Drawing
    cv2.rectangle(img, (20, 100), (50, 320), (0, 255, 0), )
    cv2.rectangle(img, (20, int(vol_bar)), (50, 320), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_per)}%', (20, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
