import datetime
import joblib
import keras
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import random as rn
import time
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
global modelPath
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from pymongo import MongoClient
import statsmodels.api as sm
from statsmodels.formula.api import ols
import math