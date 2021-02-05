# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:55:02 2021

@author: tripprakhar
"""

import os
import numpy as np
import shutil
base_path = os.path.dirname(__file__) #path to root directory
classes = ['fake', 'real'] #fake and real labels in our dataset

validation_percent = 0.15 #validation dataset is 15% of total dataset, change it acc to need
test_percent = 0.05 #test dataset is 5% of total dataset, change it acc to need

for cls in classes: #create folders and split
    os.makedirs(base_path +'/gooddata/train/' + cls) #path to train data folder
    os.makedirs(base_path +'/gooddata/validation/' + cls) #path to validation data folder
    os.makedirs(base_path +'/gooddata/test/' + cls) #path to test data folder

    src = base_path + '/saveasimages/' + cls #folder with dataset which is to be split

    filenames_all = os.listdir(src)
    np.random.shuffle(filenames_all)
    train_file_names, validation_file_names, test_file_names = np.split(np.array(filenames_all),
                                                          [int(len(filenames_all)* (1 - (validation_percent + test_percent))), 
                                                           int(len(filenames_all)* (1 - test_percent))])
    #arrays with name of images 
    train_file_names = [src+'/'+ name for name in train_file_names.tolist()]
    validation_file_names = [src+'/' + name for name in validation_file_names.tolist()]
    test_file_names = [src+'/' + name for name in test_file_names.tolist()]
    
    #print details
    print('Total ' + cls +' images: ', len(filenames_all))
    print('Training: ', len(train_file_names))
    print('Validation: ', len(validation_file_names))
    print('Testing: ', len(test_file_names))
    print('Copying!')

    # copy ans paste images from source to target folder
    for name in train_file_names:
        shutil.copy(name, base_path +'/gooddata/train/' + cls)

    for name in validation_file_names:
        shutil.copy(name, base_path +'/gooddata/validation/' + cls)

    for name in test_file_names:
        shutil.copy(name, base_path +'/gooddata/test/' + cls)