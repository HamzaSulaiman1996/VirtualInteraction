import threading
import mediapipe as mp
import cv2
import numpy as np
import pyautogui as py


class HandTrack:
    REGIONS = [4, 16, 20, 8, 12]
    MOVE_CURSOR_CONN = [1, 1, 1, 0, 0]
    LT_CLICK_CURSOR_CONN = [0, 1, 1, 0, 0]
    RT_CLICK_CURSOR_CONN = [1, 1, 1, 0, 1]
    SCROLL_START = [0, 1, 1, 1, 1]
    SCROLL_END = [0, 1, 1, 0, 1]
    prev_gesture = []
    right_prev_gesture = []
    lt_timer = 0

    def __init__(self, mindetectconf=0.5, mintrackconf=0.5):

        self.mindetectconf = mindetectconf
        self.mintrackconf = mintrackconf
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistics = mp.solutions.holistic
        self.both_hands = self.mp_holistics.Holistic(static_image_mode=False,
                                                     model_complexity=1,
                                                     smooth_landmarks=True,
                                                     enable_segmentation=False,
                                                     smooth_segmentation=True,
                                                     refine_face_landmarks=False,
                                                     min_detection_confidence=self.mindetectconf,
                                                     min_tracking_confidence=self.mintrackconf,
                                                     )
        self.width, self.height = py.size()

    def trackhands(self, frame, interact=True, draw=True):
        self.image = self.__preprocess(frame)

        cv2.putText(self.image, 'Move', (self.w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.image, 'Click', (self.w - 100, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.image, 'Scroll', (self.w - 100, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

        results = self.both_hands.process(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        if results.left_hand_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(self.image, results.left_hand_landmarks,
                                               self.mp_holistics.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec((255, 255, 51),
                                                                           thickness=1,
                                                                           circle_radius=2),
                                               )

        if results.right_hand_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(self.image, results.right_hand_landmarks,
                                               self.mp_holistics.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec((255, 255, 51),
                                                                           thickness=1,
                                                                           circle_radius=2),
                                               )

        if not interact:
            return self.image

        else:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                self.__interact(results.left_hand_landmarks, results.right_hand_landmarks)

            return self.image

    def __interact(self, right, left):
        if left:
            HandTrack.lt_timer = self.__enable_timer(HandTrack.lt_timer)
            left_handf_coordinate, bbox = self.__getxypos(left)

            cv2.putText(self.image, 'Left', (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 51),
                        1)
            cv2.rectangle(self.image, (bbox[0] - 15, bbox[1] - 15), (bbox[2] + 15, bbox[3] + 15), (255, 255, 51), 1)
            left_enc = self.__get_lm_list(left_handf_coordinate, 'left')

            xcord = int(np.mean([left_handf_coordinate[8][0], left_handf_coordinate[12][0]]))
            ycord = int(np.mean([left_handf_coordinate[8][1], left_handf_coordinate[12][1]]))
            xcord = np.interp(xcord, [0, self.w], [0, self.width + 400])
            ycord = np.interp(ycord, [0, self.h], [0, self.height + 400])

            if left_enc == HandTrack.MOVE_CURSOR_CONN:
                cv2.putText(self.image, 'Move', (self.w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                t1 = threading.Thread(target=self.__moveThread, args=[xcord, ycord])
                t1.start()

            if left_enc == HandTrack.LT_CLICK_CURSOR_CONN:
                if HandTrack.lt_timer == 0:
                    cv2.putText(self.image, 'Click', (self.w - 100, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)

                    t2 = threading.Thread(target=self.__clickThread, args=[xcord, ycord])
                    t2.start()
                    HandTrack.lt_timer = 5

            if left_enc == HandTrack.SCROLL_START:
                HandTrack.prev_gesture = HandTrack.SCROLL_START
                cv2.putText(self.image, 'Scroll', (self.w - 100, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 255, 255), 2,
                            cv2.LINE_AA)

            if (HandTrack.prev_gesture == HandTrack.SCROLL_START) and (left_enc == HandTrack.SCROLL_END):
                HandTrack.prev_gesture = []
                cv2.putText(self.image, 'Scroll', (self.w - 100, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                t3 = threading.Thread(target=self.__scrollThread, args=[-900])
                t3.start()

        if right:
            right_handf_coordinate, bbox = self.__getxypos(right)
            cv2.putText(self.image, 'Right', (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (255, 255, 51), 1)
            cv2.rectangle(self.image, (bbox[0] - 15, bbox[1] - 15), (bbox[2] + 15, bbox[3] + 15), (255, 255, 51), 1)
            right_enc = self.__get_lm_list(right_handf_coordinate, 'right')

            if right_enc == HandTrack.SCROLL_END:
                HandTrack.right_prev_gesture = HandTrack.SCROLL_END
                cv2.putText(self.image, 'Scroll', (self.w - 100, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 255, 255), 2,
                            cv2.LINE_AA)

            if (HandTrack.right_prev_gesture == HandTrack.SCROLL_END) and (right_enc == HandTrack.SCROLL_START):
                HandTrack.right_prev_gesture = []
                cv2.putText(self.image, 'Scroll', (self.w - 100, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                t4 = threading.Thread(target=self.__scrollThread, args=[900])
                t4.start()

    def __moveThread(self, xcord, ycord):
        py.moveTo(xcord, ycord, 0, py.easeOutQuad)

    def __clickThread(self, xcord, ycord):
        py.click(x=xcord, y=ycord, button='LEFT')

    def __scrollThread(self, num):
        py.vscroll(num)

    def __getxypos(self, landmarks):
        lm = {}
        fingerListx = []
        fingerListy = []
        for id, lmark in enumerate(landmarks.landmark):
            cx, cy = int(lmark.x * self.w), int(lmark.y * self.h)
            fingerListx.append(cx)
            fingerListy.append(cy)
            lm[id] = [cx, cy, lmark.z]
        bbox = [min(fingerListx), min(fingerListy), max(fingerListx), max(fingerListy)]
        return lm, bbox

    def __preprocess(self, frame):
        self.h, self.w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __enable_timer(self, timer):
        if timer > 0:
            timer -= 1
            return timer
        else:
            return 0

    def __get_lm_list(self, landmark, type=''):
        x = []
        if type == 'left':
            if landmark[HandTrack.REGIONS[0]][0] < landmark[HandTrack.REGIONS[0] - 1][0]:  # thumb closed
                x.append(1)
            else:
                x.append(0)

        elif type == 'right':
            if landmark[HandTrack.REGIONS[0]][0] > landmark[HandTrack.REGIONS[0] - 1][0]:  # thumb closed
                x.append(1)
            else:
                x.append(0)

        for reg in HandTrack.REGIONS[1:]:

            if landmark[reg][1] > landmark[reg - 2][1]:  # fingers closed
                x.append(1)
            else:
                x.append(0)
        return x