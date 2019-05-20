"""
   Setting & functions

__author__

    Alex Tselikov < atselikov@gmail.com >

"""

#imports
from __future__ import division, print_function
import pandas as pd
import numpy as np
import operator

import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from numpy.linalg import svd

from time import time
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget
from ats_functions import *
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans


############################################### vars
label2dict = {'functional':0, 'functional needs repair':1, 'non functional':2}
dict2label = {0:'functional', 1:'functional needs repair', 2:'non functional'}
dfold='data/'
sfold='subs/'
l2fold='level2/'
TARGET = 'status_group'
NFOLDS = 5
RANDOM_STATE = 2017
TOPN = 50 # to show feature importance

############################################### lists

model1_delete = ['num_private', 'wpt_name', 'extraction_type_group', 'extraction_type','payment_type', 'water_quality',\
                'scheme_management', 'district_code', 'region','region_code','subvillage', 'ward',\
                'waterpoint_type_group', 'quantity_group','installer']
cols2drop = ['id', 'recorded_by', 'quantity_group'] #recoding of quantity
    

############################################### FUNCTIONS

##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        Hidden due to competition still using for the hiring purpose
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
