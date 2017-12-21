import numpy as np
from helpers import crop_and_one_hot
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import f1_score, accuracy_score

def acc_scoring(model, imgs, gt_imgs, scoring_params={}):
    y_pred = model.predict(imgs, **scoring_params)
    y_true = np.argmax(crop_and_one_hot(gt_imgs, 16), axis=1)
    return accuracy_score(y_true, y_pred)

def f1_scoring(model, imgs, gt_imgs, scoring_params={}):
    y_pred = model.predict(imgs, **scoring_params)
    y_true = np.argmax(crop_and_one_hot(gt_imgs, 16), axis=1)
    return f1_score(y_true, y_pred)

def k_fold_model(model, imgs, gt_imgs, n_fold=4, fit_params={}, scoring_params={}, seed=42):
    cv = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    
    scoring = {'acc': lambda model, imgs, gt_imgs: acc_scoring(model, imgs, gt_imgs, scoring_params),
                   'f1': lambda model, imgs, gt_imgs: f1_scoring(model, imgs, gt_imgs, scoring_params)}
    return cross_validate(model, imgs, gt_imgs, scoring=scoring, cv=cv, fit_params=fit_params, return_train_score=False)

