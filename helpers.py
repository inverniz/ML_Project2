import os
import matplotlib.image as mpimg
import numpy as np

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

def crop_to_class(crop):
    """Assign a class to a crop according to its mean value"""
    mean = np.mean(crop)
    if mean > foreground_threshold:
        return 1
    else:
        return 0

def load_training(root_dir):
    """Load the training set must give its directory"""
    image_dir = root_dir + "/images/"
    files = os.listdir(image_dir)
    imgs = np.array([mpimg.imread(image_dir + file) for file in files])

    gt_dir = root_dir + "/groundtruth/"
    gt_imgs = np.array([mpimg.imread(gt_dir + file) for file in files])
    gt_imgs = gt_imgs.reshape((gt_imgs.shape[0], gt_imgs.shape[1], gt_imgs.shape[2], 1))
    return imgs, gt_imgs

def crop_img(img, patch, window):
    """Crop an image according to patch and window size
    If patch = window, will crop normally, otherwise, will add padding.
    """
    padding = (window - patch)// 2
    patches = []
    padded_img = np.lib.pad(img, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    for i in range(padding, img.shape[0]+padding, patch):
        for j in range(padding, img.shape[1]+padding, patch):
            img_patch = padded_img[i-padding:i+padding+patch, j-padding:j+padding+patch, :]
            patches.append(img_patch)
    return patches

def crop_imgs(imgs, patch, window):
    """"Crop a list of images into window"""
    crops = []
    for img in imgs:
        crops += crop_img(img, patch, window)
    return np.array(crops)
    
def one_hot_gt(crop):
    """One hot encode an array of groundtruth crop"""
    one_hot = [0, 0]
    label = crop_to_class(crop)
    one_hot[label] = 1
    return one_hot

def one_hot_gts(crops):
    """One hot encodes several crops"""
    num_crops = len(crops)
    one_hots = [np.zeros(2) for n in range(num_crops)]
    for i in range(num_crops):
        one_hots[i] = one_hot_gt(crops[i])
    return one_hots

def crop_and_one_hot(gts, patch=16):
    """ Crop and one hote encode the groundtruth images"""
    return np.array(one_hot_gts(crop_imgs(gts, patch, patch)))