import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import lime
import lime.lime_tabular
import dice_ml
from anchor import anchor_tabular
import shap
import matplotlib.pyplot as plt


utils_dir = './utils'
sys.path.append(utils_dir)


from explanation_utils import ExplanationUtils
import explanation_plots as ep
from DataLoader import data_loader
from importlib import reload


# Read credit data
data = data_loader()


# Load tensorflow model
model = load_model('./model/smote.tf')


# Initialize utils object
exp_utils = ExplanationUtils(model, data)



# Preprocess data
exp_utils.data_preprocess()

# Preprocess for lime and anchor
X_train_le, X_test_le, feature_names, cat_names, cat_indices = exp_utils.exp_preprocess()

X = pd.concat([X_train_le, X_test_le])



"""
SHAP
"""


predict_fn_shap = exp_utils.get_prediction_function(exp_type='shap')

shap_explainer = shap.KernelExplainer(predict_fn_shap, exp_utils.X_train)


idx=0


#file_name
file_name = os.path.join('./images/shap_' + '%02d'%idx + '.png')



shap_values = exp_utils.get_shap_exp(shap_explainer, idx, 'train', nsamples=500, l1_reg='num_features(18)', visualize=False)


ep.shap_plot(exp_utils.X_train, idx, shap_explainer, shap_values, predict_fn_shap, file_name, rename=True, num_features=6)

"""
LIME
"""

predict_fn_lime = exp_utils.get_prediction_function(exp_type='lime')
# Create explainer object, see https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular
# Open questions:
# - Kernel_width? Default is sqrt(num_features)*0.75
# - Discretize cintinuous features? If yes, can be quartile, decile and entropy
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train_le.values, feature_names=feature_names,
                                                   class_names=['Approved', 'Rejected'], mode='classification',
                                                   categorical_features=cat_indices, categorical_names=cat_names,
                                                   feature_selection='lasso_path',
                                                   discretize_continuous=True)


lime_explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=feature_names,
                                                       class_names=['Approved', 'Rejected'], mode='classification',
                                                       categorical_features=cat_indices, categorical_names=cat_names,
                                                       feature_selection='lasso_path',
                                                       discretize_continuous=True)


idx = 0


lime_exp = exp_utils.get_lime_exp(lime_explainer, predict_fn_lime, idx, 'train', num_features=18)

#file_name
file_name = os.path.join('./images/lime_' + '%02d'%idx + '.png')


ep.lime_plot(exp_utils.X_train, idx, lime_exp, file_name, rename=True, num_features=6)




"""
Anchors
"""


anchor_explainer = anchor_tabular.AnchorTabularExplainer(
    ['Approved', 'Rejected'],
    feature_names,
    X_train_le.values,
    cat_names,
    discretizer='quartile')

# Prediction function
predict_fn_anchor = exp_utils.get_prediction_function('anchor')


idx = 0


anchor_exp, anchor_proba = exp_utils.get_anchor_exp(anchor_explainer, predict_fn_anchor, predict_fn_lime, idx, 'train', plot_mode=True, threshold=0.9)                                             


#file_name
file_name = os.path.join('./images/anchors_' + '%02d'%idx + '.png')


ep.anchor_plot(exp_utils.X_train, idx, anchor_exp, anchor_proba, file_name, rename=True)


"""
DICE
"""



# Create data object. Specify the continours features and outcome variable
d = dice_ml.Data(dataframe=exp_utils.train_data, continuous_features=['Duration (months)', 'Amount (EUR)', 'Age (years)'], outcome_name='label')
# Provide the trained ML model to DiCE's model object
backend = 'TF'+tf.__version__[0]
m = dice_ml.Model(model=exp_utils.model, backend=backend)
# Create explainer object with trained model and data object
dice_explainer = dice_ml.Dice(d, m)

idx=0


dice_exp = exp_utils.get_dice_exp(dice_explainer, idx, 'train', total_CFs=3, desired_class='opposite')

#file_name
file_name = os.path.join('./images/dice_' + '%02d'%idx + '.png')


ep.dice_plot(exp_utils.X_train, idx, dice_exp, file_name, version='0.5', fill_empty='-', fill_cell=True, alignment='center', rename=True)


