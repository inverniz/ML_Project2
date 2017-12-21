import numpy as np

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from keras.applications.vgg16 import VGG16

class FCNModel:
    """Fully convolutional model model that can take any image size as input
    and predicts road presence for patches of size 16x16"""
    
    def __init__(self, **args):
        pass
    
    def initialize_model(self, dropout=False, weights=None):
        """Create the model from anew"""
        
        # Reset the tensorflow backend to have an empty GPU
        K.clear_session()
        
        # Input can have arbitraty size
        inputs = Input((None,None,3))
        
        # If weights='imagenet', will automatically download them
        vgg16 = VGG16(include_top=False, weights=weights, input_tensor=inputs, input_shape=(None, None, 3))
        
        # Actual model, cf. paper
        input_vgg16 = vgg16.get_layer('block3_pool').output
        if dropout:
            input_vgg16 = Dropout(0.5)(input_vgg16)
        output = UpSampling2D(2)(input_vgg16)
        output = BatchNormalization()(Conv2D(256, 3, padding='same', activation='relu')(output))
        output = BatchNormalization()(Conv2D(256, 3, padding='same', activation='relu')(output))
        output = BatchNormalization()(Conv2D(256, 3, padding='same', activation='relu')(output))
        if dropout:
             output = Dropout(0.25)(output)
        output = UpSampling2D(2)(output)
        output = BatchNormalization()(Conv2D(128, 3, padding='same', activation='relu')(output))
        output = BatchNormalization()(Conv2D(128, 3, padding='same', activation='relu')(output))
        if dropout:
             output = Dropout(0.25)(output)
        output = UpSampling2D(2)(output)
        output = BatchNormalization()(Conv2D(64, 3, padding='same', activation='relu')(output))
        output = BatchNormalization()(Conv2D(64, 3, padding='same', activation='relu')(output))
        if dropout:
             output = Dropout(0.25)(output)
        output = Conv2D(2, 1, padding='same', activation='softmax')(output)

        self.model = Model(inputs, output)

    
    def fit(self, imgs, gt_imgs, augment=False, dropout = False, weights=None, restore=False,
            num_batches=2, steps_per_epoch=64, epochs=5, validation_steps=32, verbose = 1, seed=42):
        """Fit model completely parametrizable, receives images and groundtruth
        Can define the usage or not of a dropout and of pre-trained weights (None or 'imagenet')
        Can also choose parameters for the training"""

        self.initialize_model(dropout, weights)
        
        if restore:
            self.model = load_model('fcn.h5')
            return

        def one_hot_encode(gt):
            """One hot encode an image without cropping it"""
            # We need values that are either 0 or 1
            gt = np.round(gt).astype(int)
            # The final shape of the encoding
            encoded = np.zeros((gt.shape[0], gt.shape[1], 2))
            for i in range(gt.shape[0]):
                for j in range(gt.shape[1]):
                    encoded[i][j][gt[i][j]] = 1
            return encoded
        
        def generate_train(batch_size, imgs, gt_imgs, augment=False, seed=1):
            """Training batches generator, generates batches of num_batches
            with the possibility to augment the images"""
            
            np.random.seed(seed)
            while 1:
                batch_x = []
                batch_y = []
                for i in range(batch_size):
                    # We simply pick a random image
                    idx = np.random.choice(imgs.shape[0])
                    img = imgs[idx]
                    gt = one_hot_encode(gt_imgs[idx])
                    # Augment if asked
                    if augment:
                        if np.random.choice(2):
                            img_crop = np.flipud(img)
                            gt_crop = np.flipud(gt)
                        if np.random.choice(2):
                            img_crop = np.fliplr(img)
                            gt_crop = np.fliplr(gt)
                        rot = np.random.choice(4)
                        img_crop = np.rot90(img, rot)
                        gt_crop = np.rot90(gt, rot)
                    batch_x.append(img)
                    batch_y.append(gt)
                yield(np.array(batch_x), np.array(batch_y))
        
        train_generator = generate_train(num_batches, imgs, gt_imgs, augment=augment, seed=seed)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose)
        
    def predict(self, imgs, batch_size=2):
        """Predict the features of the given images
        the return format is a vector of shape #images x image_width x image_height
        possibility to precise the batch size if the memory is limited.
        Can also restore from a pre-trained model"""
        
        predictions = self.model.predict(imgs, batch_size)
        # Format of prediction is (num_image, img_size, img_size, 2)
        # we first find if road or background with an argmax, giving us
        # (num_image, img_size, img_size)
        predictions = np.argmax(predictions, axis=3)
        # Then we reshape it to fit the format wanted by crop_and_one_hot
        # i.e. (num_image, img_size, img_size, 1)
        predictions = np.expand_dims(predictions, -1)
        # Finally, we transform it in the format
        # (num_image * img_size//patch_size * img_size//patch_size)
        return np.argmax(crop_and_one_hot(predictions), axis=1)
    
    def get_params(self, deep=True):
        """Obligatory function for cross validation of sklearn, not used"""
        return {'x': None}
    
    def set_params(self, **params):
        """Obligatory function for cross validation of sklearn, not used"""
        pass