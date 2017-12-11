# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility

import math
import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K

WINDOW_SIZE = 80
IMG_PATCH_SIZE = 16
TRAINING_SIZE = 100
TEST_SIZE = 50
PADDING_SIZE = math.ceil((WINDOW_SIZE - IMG_PATCH_SIZE)/2)

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract windows from a given image
def img_window(im, w, h):
    list_windows = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(PADDING_SIZE,imgheight - PADDING_SIZE,h):
        for j in range(PADDING_SIZE,imgwidth - PADDING_SIZE,w):
            if is_2d:
                im_window = im[j-PADDING_SIZE:j+w+PADDING_SIZE, i-PADDING_SIZE:i+h+PADDING_SIZE]
            else:
                im_window = im[j-PADDING_SIZE:j+w+PADDING_SIZE, i-PADDING_SIZE:i+h+PADDING_SIZE, :]
            list_windows.append(im_window)            
    return list_windows

def extract_data_training(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    
    print("number of images: " + str(num_images))
    
    #IMG_WIDTH = imgs[0].shape[0]
   # IMG_HEIGHT = imgs[0].shape[1]
    
    print("image shape: " +  str(imgs[0].shape))
    
    #N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], WINDOW_SIZE, WINDOW_SIZE) for i in range(num_images)]
    
    patches_data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    
    print("number of patches: " + str(len(patches_data)))
    print("patch shape: " + str(patches_data[0].shape))
    print(numpy.asarray(patches_data).shape)
    return numpy.asarray(patches_data)

def extract_data_testing(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    
    imgs = []
    padded_imgs = []
    for i in range(1, num_images+1):
        dirid = "test_" + i + "/"
        imageid = "test_" + i
        image_filename = filename + dirid + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            padded_img = np.lib.pad(img, ((PADDING_SIZE, PADDING_SIZE), (PADDING_SIZE, PADDING_SIZE), (0,0)), 'reflect')
            imgs.append(img)
            padded_imgs.append(padded_img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    
    print("number of images: " + str(num_images))
    
    #IMG_WIDTH = imgs[0].shape[0]
    #IMG_HEIGHT = imgs[0].shape[1]
    
    print("image shape: " +  str(imgs[0].shape))
    print("padded image shape: " +  str(padded_imgs[0].shape))
    
    #N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    img_windows = [img_window(padded_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    
    patches_data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    windows_data = [img_windows[i][j] for i in range(len(img_windows)) for j in range(len(img_windows[i]))]
    
    print("number of patches: " + str(len(patches_data)))
    print("number of windows: " + str(len(windows_data)))
    print("patch shape: " + str(patches_data[0].shape))
    print("window shape: " + str(windows_data[0].shape))
    
    return numpy.asarray(windows_data),  numpy.asarray(patches_data)

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], WINDOW_SIZE, WINDOW_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

def main():
    
    # needed for some reason
    K.set_image_dim_ordering('th')
    
    
    #obtain training images
    train_data_dir = 'training/'
    train_data_filename = train_data_dir + 'images/'
    train_labels_filename = train_data_dir + 'groundtruth/' 

    train_data = extract_data_training(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)
    
    #obtain test_images
    test_data_dir = 'test_set_images/'
    
    test_data, patches = (test_data_dir, TEST_SIZE)
    
    #preprocessing
    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print (len(new_indices))
    print (train_data.shape)
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

     #model 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80,80,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    # 9. Fit model on training data
    model.fit(train_data, train_labels, 
          batch_size=32, epochs=10, verbose=1)
    # 10. Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
    

if __name__ == "__main__":
	main()
