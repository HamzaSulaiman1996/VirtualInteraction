import cv2
from holistics import HandTrack
import pyautogui as py
import time
import argparse


def main(args):
    handtrack = HandTrack(mindetectconf=args.mindetectconf, mintrackconf=args.mintrackconf)
    py.FAILSAFE = False
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    fps_start_time = 0

    while cap.isOpened():

        ret, frame = cap.read()
        frame = cv2.resize(frame, (args.resolution[0], args.resolution[1]), interpolation=cv2.INTER_AREA)
        frame = handtrack.trackhands(frame=frame, interact=args.interact, draw=True)

        fps_end_time = time.time()
        fps_diff = fps_end_time - fps_start_time
        fps = 1 / fps_diff
        fps_start_time = fps_end_time

        cv2.putText(frame, str(round(fps, 2)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interact", help="Enable interaction. DEFAULT=TRUE",
                        action="store_true")
    parser.add_argument("-cd", "--mindetectconf", help="Minimum detection confidence DEFAULT=0.3", type=float,
                        default=0.3)
    parser.add_argument("-ct", "--mintrackconf", help="Minimum tracking confidence DEFAULT=0.3", type=float,
                        default=0.3)

    parser.add_argument("-r", "--resolution", help="Feed Resolution DEFAULT=640X480", nargs=2, type=int,
                        default=[640, 480])

    args = parser.parse_args()
    main(args)
