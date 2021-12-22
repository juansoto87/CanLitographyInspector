import RPi.GPIO as GPIO
from time import *
from Create_folder import create_folder
import cv2
import os


CF = create_folder()
CF.create_new_folder('Club_N2_c1')
folder = CF.new_folder
print(folder)
GPIO.cleanup()

control = 12

GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.OUT)

frame_rate = 8
ON = True
t1 = time()

cap = cv2.VideoCapture(1)
#ret, frame = cap.read()
sleep(2)
while ON:
    print('Inicio')
    GPIO.output(control, GPIO.HIGH)
    number = 1
    while number < 40:
        t3 = time()
        ret, frame = cap.read()
        print(f'Foto: {number}')
        #cv2.imwrite(f'mypicture_{number}.png',frame)
        name = f'mypicture_{number}.png'
        cv2.imwrite(os.path.join(folder, name),frame)
        number += 1
        t4 = time()
        print(f'Tiempo entre fotos: {t4 - t3}')
        #sleep(1/frame_rate)
    
    GPIO.output(control, GPIO.LOW)
    ON = False
    cap.release()
    cv2.destroyAllWindows()
    
t2 = time()

print(f'Tiempo total: {t2 - t1}')
GPIO.cleanup()
print('Terminado')