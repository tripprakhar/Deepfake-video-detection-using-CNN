# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:55:48 2021

@author: tripprakhar
"""

import cv2
import glob

fake_dir = 'traindata/fake/'#path to cropped fake videos folder, change it according to your path
real_dir = 'traindata/real/'#path to cropped real videos folder, change it according to your path
target_fake_dir = 'imagetrain/fake/'#path to cropped fake images folder, change it according to your path
target_real_dir = 'imagetrain/real/'#path to cropped real images folder, change it according to your path
t_size = (128,128)
fake_video_files = glob.glob(fake_dir + '*.mov')
real_video_files = glob.glob(real_dir + '*.mov')
print('Number of fake video files: ', len(fake_video_files))
print('Number of real video files: ', len(real_video_files))
    
def generate_face_videos(source_files, target_dir, t_size):
    i = 1
    for filename in source_files:
        cap= cv2.VideoCapture(filename)
        
        ret, frame = cap.read()
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            print("0")
            out_name = target_dir + '{0:03d}'.format(i) + '.png'
            i+=1
            cv2.imwrite(out_name,frame)
            
        
generate_face_videos(fake_video_files, target_fake_dir, t_size)
generate_face_videos(real_video_files, target_real_dir, t_size)