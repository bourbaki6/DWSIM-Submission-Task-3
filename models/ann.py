

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
import os
import time
import warnings

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)
 
SEED = 42
np.random.seed(SEED)
 
TARGET_NAMES = ["xD", "xB", "QC_kW", "QR_kW"]


class ANN:

    def train_ann(X_tr, y_tr_s, X_val, y_val_s, X_te, y_te, scaler_y):
        
        print("Model 4: Artificial Neural Network (MLP, Keras)")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, regularizers, callbacks
    
        except ImportError:
            print("TensorFlow not available. Skipping ANN.")
        
            return None
     
 
        tf.random.set_seed(SEED)
        n_features = X_tr.shape[1]
        n_outputs  = y_tr_s.shape[1]
 
        def build_model(lr = 5e-4, dropout = 0.1, l2 = 1e-4):
            
            inp = keras.Input(shape = (n_features,))
            x = layers.Dense(128, activation = "relu", kernel_regularizer = regularizers.l2(l2))(inp)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Dense(64, activation="relu", kernel_regularizer = regularizers.l2(l2))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Dense(32, activation = "relu")(x)
            out = layers.Dense(n_outputs, activation = "linear")(x)
            model = keras.Model(inp, out)
            model.compile(optimizer = keras.optimizers.Adam(lr), loss = "mse", metrics = ["mae"],
            )
            return model
 
    
    