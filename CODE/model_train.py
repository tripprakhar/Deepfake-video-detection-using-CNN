# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 19:59:56 2021

@author: tripprakhar
"""

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'gooddata/train' #path to train dataset
val_dir = 'gooddata/validation' #path to validation dataset

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=(128, 128),  
        batch_size=256,
        class_mode='binary')

validation_generator = valid_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=256,
        class_mode='binary')

#class myCallback(tf.keras.callbacks.Callback):
    #def on_epoch_end(self, epoch, logs={}):
        #if(logs.get('acc')>0.99):
            #print("\n 99% accuracy")
            #self.model.stop_training = True
            
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0010,  patience=5)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(
      train_generator,
      epochs=20,
      validation_data=validation_generator,
      callbacks = [callbacks],
      #max_queue_size=32,
      #workers = 8,
      verbose=1)

model.save('trained_native.h5')

#model = tf.keras.models.load_model('trained_native.h5') #uncomment to load model from above listed file
test_dir = 'gooddata/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=256,
        class_mode='binary')

print('\n# Evaluate on test data')
results = model.evaluate(test_generator)
print('test loss, test acc:', results)
