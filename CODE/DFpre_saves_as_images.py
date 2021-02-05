# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 07:11:29 2020

@author: tripprakhar
"""
import cv2
#import face_recognition
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
input_movie = cv2.VideoCapture(r'D:/deepfake/DFdata/cxttmymlbn.mp4',0)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0

while True:
    #grab frames
    ret, frame = input_movie.read()
    frame_number += 1

    #end when video ends
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        roi_color = frame[y:y+h, x:x+w]

    img_item=str(frame_number)+".png"
    cv2.imwrite(img_item, roi_color)
    print("saving frame {} / {}".format(frame_number, length))


input_movie.release()
cv2.destroyAllWindows()
