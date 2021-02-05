
"""
Created on Mon Feb  1 18:15:37 2021

@author: tripprakhar
"""

import cv2
import glob

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml') #cv2 cascade for face detection
fake_dir = 'sortedvid/fake/' #path to fake videos folder,change it according to your path
real_dir = 'sortedvid/real/' #path to real videos folder,change it according to your path
target_fake_dir = 'saveasimages/fake/' #path to cropped fake images folder,change it according to your path
target_real_dir = 'saveasimages/real/'#path to cropped real images folder,change it according to your path
fake_video_files = glob.glob(fake_dir + '*.mp4')
real_video_files = glob.glob(real_dir + '*.mp4')
print('Number of fake video files: ', len(fake_video_files))
print('Number of real video files: ', len(real_video_files))

    
def generate_face_images(source_files, target_dir):
    i = 1
    for filename in source_files:
        cap= cv2.VideoCapture(filename)
        count = 0
        ret, frame = cap.read()
        height, width, layers = frame.shape
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break           
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
            for(x,y,w,h) in faces:
                cropped_face = frame[y:y+h, x:x+w]            
                out_name = target_dir + '{0:03d}'.format(i) + '.png'
                cv2.imwrite(out_name,cropped_face)
                i+=1
            count += 10 #at 30fps this means we are taking 3 frames per sec
            cap.set(1, count)
        
generate_face_images(fake_video_files, target_fake_dir)
generate_face_images(real_video_files, target_real_dir)