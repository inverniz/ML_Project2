import numpy as np

class DummyModel:
    """A dummy model that predicts a truth with 20% of roads"""
    def __init__(self, **args):
        pass
    
    def fit(self, imgs, gt_imgs):
        pass
        
    def predict(self, imgs):
        # Fix the seed 
        np.random.seed(42)
        return (np.random.rand((imgs.shape[0]*imgs.shape[1]//16*imgs.shape[2]//16)) > 0.8).astype(int)
    
    def get_params(self, deep=True):
        return {'x': None}
    
    def set_params(self, **params):
        pass