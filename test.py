import cv2
import numpy as np
from holistics import HandTrack
import pyautogui as py
import time

handtrack = HandTrack(mindetectconf=0.3, mintrackconf=0.3)
width, height = py.size()

py.FAILSAFE = False

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
TIMER = 0
fps_start_time = 0
fps = 0
while cap.isOpened():

    ret, frame = cap.read()

    frame = handtrack.trackhands(frame=frame, interact=True, draw=True)

    fps_end_time = time.time()
    fps_diff = fps_end_time - fps_start_time
    fps = 1 / fps_diff
    fps_start_time = fps_end_time

    cv2.putText(frame, str(round(fps, 2)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
