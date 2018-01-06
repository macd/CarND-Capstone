# This needs keras > 2.0 (I think).  I ran OK on keras 2.1.2
import numpy as np
import os, sys

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint

# dimensions of all images will be resized to the following
img_width, img_height = 300, 400

train_data_dir = 'data/train'

# We really don't have the luxury of real validation data
validation_data_dir = 'data/train'

nb_train_samples = 1161
nb_validation_samples = 1161
epochs = 200
batch_size = 16


input_shape = (img_width, img_height, 3)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax'))

    model.summary()
    return model

def train_model(model, callbacks=[]):
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(    
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks)


if __name__ == "__main__":
    model = build_model()
    cb = ModelCheckpoint('best_model.hd5',
                    monitor='val_loss',
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=False,
                    mode='auto',
                    period=1)
    train_model(model, [cb])
    model.model.save('track_model.hd5')    
