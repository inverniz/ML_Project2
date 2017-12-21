import numpy as np
import numpy.random as random

from helpers import *

import keras.models as models
import keras.backend as K
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from sklearn.utils import class_weight
from keras.regularizers import l2

class CNNModel:
    """Convolutional model that predict patches of 16x16 pixels
    from window that can have size from 16 to 400 pixels"""
    
    def __init__(self, **args):
        pass
    
    def initialize_model(self, window=72, dropout=False):
        """Create the model from anew"""
        # Reset the tensorflow backend to have an empty GPU
        K.clear_session()
        
        # Actul Model, cf. report
        self.model = models.Sequential()
        self.model.add(Conv2D(32, 5, padding='same', activation='relu', input_shape=(window,window,3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        if dropout:
            self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, 3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        if dropout:
            self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, 3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, 3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        if dropout:
            self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        if dropout:
            self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
    
    def fit(self, imgs, gt_imgs, patch=16, window=72, augment=False, dropout = False,
            num_batches=128, steps_per_epoch=64, epochs=5, validation_steps=32, seed=42, verbose=1):
        """Fit model completely parametrizable, receives images and groundtruth
        Can define the patch size, the window size, if the data has to be augmented
        and if a dropout has to be used. Can also choose parameters for the training"""
        
        # Reset the model
        self.initialize_model(window, dropout)
        self.window = window
        
        def random_crop(img, gt_img, patch=16, window=72, seed=1):
            """Randomly crop an image and its groudtruth returns an image of size
            window x window, center of window is around a patch and can be in the
            corner since padding is added"""
            
            # Adding padding
            padding = (window-patch)//2
            img = np.lib.pad(img, ((padding, padding), (padding, padding), (0,0)), 'reflect')
            gt_img = np.lib.pad(gt_img, ((padding, padding), (padding, padding), (0,0)), 'reflect')

            # Randomly picking a center that correspond to a patch
            center = np.random.randint(window//2, img.shape[0] - window//2, 2)

            # Crop the images
            img_crop = img[center[0]-window//2:center[0]+window//2, center[1]-window//2:center[1]+window//2]
            gt_crop = gt_img[center[0]-patch//2:center[0]+patch//2, center[1]-patch//2:center[1]+patch//2]

            # Return the images, groudtruth one hot encoded
            return img_crop, one_hot_gt(gt_crop)

        def generate_train(num_batches, imgs, gt_imgs, patch=16, window=72, augment=False, seed=1):
            """Training batches generator, generates batches of num_batches of window
            with the possibility to augment the images"""
            while 1:
                batch_x = []
                batch_y = []
                for i in range(num_batches):
                    # Randomly pick an image
                    idx = random.choice(imgs.shape[0])
                    # Randomly pick a crop
                    train_x, train_y = random_crop(imgs[idx], gt_imgs[idx], patch, window, seed)
                    # Randomly flip and rotate
                    if augment:
                        if random.choice(2):
                            img_crop = np.flipud(train_x)
                        if random.choice(2):
                            img_crop = np.fliplr(train_x)
                        rot = np.random.choice(4)
                        img_crop = np.rot90(train_x, rot)

                    batch_x.append(train_x)
                    batch_y.append(train_y)
                yield(np.array(batch_x), np.array(batch_y))
                
        train_generator = generate_train(num_batches, imgs, gt_imgs, patch=patch, window=window,
                                         augment=augment, seed=seed)

        # Compute the class weight on the given training set
        cw = class_weight.compute_class_weight('balanced', [0,1], np.argmax(crop_and_one_hot(gt_imgs, 16), axis=1))
        cw = dict(enumerate(cw))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                 class_weight=cw, verbose=verbose)#, validation_data = (X_test, y_test))
    
    def predict(self, imgs, batch_size=128):
        """"Predict the features of the given images
        the return format is a vector of shape #images x image_width x image_height
        possibility to precise the batch size if the memory is limited"""
        cropped_imgs = crop_imgs(imgs, 16, self.window)
        predictions = self.model.predict(cropped_imgs, batch_size)
        return np.argmax(predictions, axis=1)
    
    def get_params(self, deep=True):
        """Obligatory function for cross validation of sklearn, not used"""
        return {'x': None}
    
    def set_params(self, **params):
        """Obligatory function for cross validation of sklearn, not used"""
        pass