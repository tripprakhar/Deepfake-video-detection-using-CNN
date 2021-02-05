# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:15:37 2021

@author: tripprakhar
"""



import cv2
import glob

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml') #cv2 cascade for face detection
fake_dir = 'sortedvid/fake/'  #path to fake videos, change it according to your path
real_dir = 'sortedvid/real/'  #path to real videos, change it according to your path
target_fake_dir = 'face_videos/fake1/'   #path to cropped fake videos folder, change it according to your path
target_real_dir = 'face_videos/real1/'   #path to cropped real videos folder,change it according to your path
t_size = (128,128)
fake_video_files = glob.glob(fake_dir + '*.mp4')
real_video_files = glob.glob(real_dir + '*.mp4')
print('Number of fake video files: ', len(fake_video_files))
print('Number of real video files: ', len(real_video_files))

def get_cropped_face(frame = None, r_size = (128,128), s_size = None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    roi_color=0
    for(x,y,w,h) in faces:
        roi_color = frame[y:y+h, x:x+w]
    resized_face = cv2.resize(roi_color, r_size)
    return resized_face
    
def generate_face_videos(source_files, target_dir, t_size):
    i = 1
    for filename in source_files:
        cap= cv2.VideoCapture(filename)
        img_array = []
        count = 0
        ret, frame = cap.read()
        height, width, layers = frame.shape
        s_size = (width, height)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            print(0)
            cropped_face = get_cropped_face(frame, t_size, s_size)
            print(1)
            
            if cropped_face is not None:
                img_array.append(cropped_face)
            count += 12 #at 30fps this means we are taking 3 frames per sec
            cap.set(1, count)
        
        out_name = target_dir + '{0:03d}'.format(i) + '.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_name,fourcc, 2, t_size)
        for j in range(len(img_array)):
            out.write(img_array[j])
        out.release()
        i += 1
        
generate_face_videos(fake_video_files, target_fake_dir, t_size)
generate_face_videos(real_video_files, target_real_dir, t_size)