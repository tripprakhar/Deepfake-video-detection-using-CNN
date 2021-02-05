# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:15:37 2021

@author: tripprakhar
"""

import cv2
import glob
import os

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml') #cv2 cascade for face detection
test_dir = 'test_videos/'  #path to videos, change it according to your path
target_dir = 'test_videos_cropped/'   #path to cropped videos folder, change it according to your path
t_size = (128,128)
video_files = glob.glob(test_dir + '*.mp4')
print('Number of video files: ', len(video_files))

    
def generate_face_videos(source_files, target_dir, t_size):
    i = 1
    for filename in source_files:
        cap= cv2.VideoCapture(filename)
        count = 0
        ret, frame = cap.read()
        height, width, layers = frame.shape
        img_array=[]
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break           
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
            for(x,y,w,h) in faces:
                cropped_face = frame[y:y+h, x:x+w]   
                cropped_face = cv2.resize(cropped_face, t_size)
                img_array.append(cropped_face)
            count += 10 #at 30fps this means we are taking 3 frames per sec
            cap.set(1, count)
        
        name = os.path.basename(filename)
        out_name = target_dir + name[:-4] + '.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_name,fourcc, 2, t_size)
        for j in range(len(img_array)):
            out.write(img_array[j])
        out.release()
        i += 1
        
generate_face_videos(video_files, target_dir, t_size)