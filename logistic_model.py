from helpers import *

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

class LogisticModel:
    """Logistic model encoded as a basic class that allow to easily use cross validation"""
    def __init__(self, **args):
        """Initialization of the model"""
        self.patch_size = 16 # Size of a patch 16x16 pixels
        self.poly_aug = 5 # Degree of the polynomical augmentation
        self.logreg = LogisticRegression(C=1e5, class_weight="balanced")
    
    def extract_features(self, img):
        """Extract 6-dimensional features consisting of average RGB color as well as variance"""
        feat_m = np.mean(img, axis=(0,1))
        feat_v = np.var(img, axis=(0,1))
        feat = np.append(feat_m, feat_v)
        return feat
    
    def fit(self, imgs, gt_imgs):
        """Fit the given imgs and groundtruth in the model"""
        
        # Crop the image in patches of 16x16
        img_patches = crop_imgs(imgs, self.patch_size, self.patch_size)
        gt_patches = crop_imgs(gt_imgs, self.patch_size, self.patch_size)
        
        # Extract the features of the patches
        X = np.asarray([self.extract_features(img_patches[i]) for i in range(len(img_patches))])
        # Make the ground-truth patches to 0-1
        Y = np.asarray([crop_to_class(gt_patches[i]) for i in range(len(gt_patches))])
        # Augment the features to a polynomial
        poly = PolynomialFeatures(self.poly_aug, interaction_only=False)
        X = poly.fit_transform(X)
        self.logreg.fit(X, Y)
        
    def predict(self, imgs):
        """"Predict the features of the given images
        the return format is a vector of shape #images x image_width x image_height"""
        
        img_patches = crop_imgs(imgs, self.patch_size, self.patch_size)
        X = np.asarray([self.extract_features(img_patches[i]) for i in range(len(img_patches))])
        poly = PolynomialFeatures(self.poly_aug, interaction_only=False)
        X = poly.fit_transform(X)
        return self.logreg.predict(X)
    
    def get_params(self, deep=True):
        """Obligatory function for cross validation of sklearn, not used"""
        return self.logreg.get_params(deep)
    
    def set_params(self, **params):
        """Obligatory function for cross validation of sklearn, not used"""
        return self.logreg.set_params(params)