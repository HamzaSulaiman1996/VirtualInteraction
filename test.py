import cv2
import numpy as np
import holistics
from holistics import HandTrack
import pyautogui as py
import time

handtrack = HandTrack(mindetectconf=0.7, mintrackconf=0.7)
width, height = py.size()

py.FAILSAFE = False

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()

    right, left, frame = handtrack.trackhands(image=frame, draw=True)
    cv2.putText(frame, 'Move', (551, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Click', (551, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    left_lmark = handtrack.get_landmarks(left)
    right_lmark = handtrack.get_landmarks(right)

    if left_lmark:
        x = []
        if left_lmark[holistics.REGIONS[0]][0] < left_lmark[holistics.REGIONS[0] - 1][0]:  # thumb closed
            x.append(1)
        else:
            x.append(0)

        for reg in holistics.REGIONS[1:]:

            if left_lmark[reg][1] > left_lmark[reg - 2][1]:  # fingers closed
                x.append(1)
            else:
                x.append(0)

        if x == holistics.MOVE_CURSOR_CONN:
            cv2.putText(frame, 'Move', (551, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            xcord = np.mean([left_lmark[8][0], left_lmark[12][0]])
            ycord = np.mean([left_lmark[8][1], left_lmark[12][1]])
            xcord = np.interp(xcord, [0, frame.shape[1]], [0, width + 400])
            ycord = np.interp(ycord, [0, frame.shape[0]], [0, height + 400])
            py.moveTo(xcord, ycord, duration=0)

        if x == holistics.CLICK_CURSOR_CONN:
            cv2.putText(frame, 'Click', (551, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            py.click()
            time.sleep(1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
