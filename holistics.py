import mediapipe as mp
import cv2

REGIONS = [4, 16, 20, 8, 12]
MOVE_CURSOR_CONN = [1, 1, 1, 0, 0]
CLICK_CURSOR_CONN = [0, 1, 1, 0, 0]


class HandTrack:
    def __init__(self,
                 mindetectconf=0.5,
                 mintrackconf=0.5):

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

    def trackhands(self, image, draw=True):
        self.h, self.w, _ = image.shape
        left = None
        right = None
        image = cv2.flip(image, 1)  ## cause left hand to be right hand and vice versa
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.both_hands.process(frame)

        if results.left_hand_landmarks:

            left = results.left_hand_landmarks
            if draw:
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistics.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec((255, 255, 51)),
                                               )

        if results.right_hand_landmarks:
            right = results.right_hand_landmarks

            if draw:
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistics.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec((255, 255, 51)),
                                               )

        return left, right, image

    def get_landmarks(self, handlandmark):
        if handlandmark is not None:
            lm = {}
            for id, lmark in enumerate(handlandmark.landmark):
                cx, cy = int(lmark.x * self.w), int(lmark.y * self.h)
                lm[id] = [cx, cy, lmark.z]
            return lm

