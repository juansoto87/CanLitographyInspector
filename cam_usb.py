import numpy as np
import cv2
import os

#cap = cv2.VideoCapture(1)

def take_picture(number, path):
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(path, f'mypicture_{number}.png'),frame)
    cap.release()
    cv2.destroyAllWindows()

def cal_cam():
    cap = cv2.VideoCapture(1)
    while(True):
        ret, frame = cap.read()
        #cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
        #cv2.resize(frame, (960, 540))
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

cal_cam()

