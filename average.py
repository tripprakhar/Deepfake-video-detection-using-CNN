# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:05:16 2021

@author: tripprakhar
"""

import cv2
import glob
import numpy as np
import tensorflow as tf

dire = 'test_videos/'
video_files = glob.glob(dire + '*.mp4')

def get_video_prediction(filename, model):
    cap= cv2.VideoCapture(filename)
    img_array = []
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if (ret == False) or (count > num_frames):
            break
        img_array.append(tf.convert_to_tensor(frame))
        count += 1 
    if img_array:
        img_batch = tf.stack(img_array)    
        val = model.predict(img_batch)
        return np.round(np.mean(val))

model  = tf.keras.models.load_model('trained_native.h5')
success = 0
for filename in video_files:
    pred = get_video_prediction(filename, model)
    print(pred)

