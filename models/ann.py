
 
from __future__ import annotations
import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from models.base import (ModelResult, TARGET_NAMES,
                               print_metrics, compute_metrics, SEED)
 
DEFAULT_PATH = "models/ann_model.keras"


def _build_keras_model(n_features: int, n_outputs: int,
                       lr: float, dropout: float, l2: float):
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    tf.random.set_seed(SEED)
 
    inp = keras.Input(shape = (n_features,), name = "inputs")

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name="dense_1")(inp)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout, name="drop_1")(x)
 
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout, name="drop_2")(x)
 
    x = layers.Dense(32, activation="relu", name="dense_3")(x)
 
    out = layers.Dense(n_outputs, activation="linear", name="outputs")(x)
 
    model = keras.Model(inp, out, name="distillation_surrogate_ann")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    
    return model
 
def _make_predict_fn(keras_model, scaler_y):
    
    def predict_fn(X_scaled: np.ndarray) -> np.ndarray:
        
        y_scaled = keras_model.predict(X_scaled, verbose=0)
        
        return scaler_y.inverse_transform(y_scaled)
    
    return predict_fn

def _plot_training_history(history, save_dir: str = "plots") -> None:

    os.makedirs(save_dir, exist_ok = True)
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))

    axes[0].plot(history.history["loss"], label = "Train loss")
    axes[0].plot(history.history["val_loss"], label = "Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (scaled)")
    axes[0].set_title("ANN — training loss curve")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)
 
    axes[1].plot(history.history["mae"], label = "Train MAE")
    axes[1].plot(history.history["val_mae"], label = "Val MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (scaled)")
    axes[1].set_title("ANN — MAE curve")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)
 
    plt.tight_layout()
    path = os.path.join(save_dir, "ann_training_curves.png")
    plt.savefig(path, dpi = 150, bbox_inches = "tight")
    plt.close()
    
    print(f" Saved: {path}")

def train(
    X_tr:    np.ndarray,
    y_tr_s:  np.ndarray,     # SCALED targets
    X_val:   np.ndarray,
    y_val_s: np.ndarray,     # SCALED targets
    y_val:   np.ndarray,     # UNSCALED targets (for final metric)
    scaler_y,                # fitted StandardScaler for targets
    lr:       float = 5e-4,
    dropout:  float = 0.10,
    l2:       float = 1e-4,
    epochs:   int   = 500,
    batch_size: int = 32,
    patience:   int = 30,
    save_path:  str = DEFAULT_PATH,
) -> ModelResult:
    
    try:
        import tensorflow as tf
        from tensorflow.keras import callbacks
    
    except ImportError:
        print(" [ANN] TensorFlow is not installed. Skipping ANN training.")
        print(" Install with:  pip install tensorflow")
        
        return None
    
    print("  Model: Artificial Neural Network (MLP, Keras)")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
 
    n_features = X_tr.shape[1]
    n_outputs = y_tr_s.shape[1]
 
    keras_model = _build_keras_model(n_features, n_outputs, lr, dropout, l2)
    keras_model.summary(print_fn = lambda x: print(f" {x}"))
 
    os.makedirs("models", exist_ok = True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = patience,
            restore_best_weights = True,
            verbose = 1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor = "val_loss",
            factor = 0.5,
            patience = patience // 3,
            min_lr = 1e-6,
            verbose = 0,
        ),
        callbacks.ModelCheckpoint(
            save_path,
            monitor = "val_loss",
            save_best_only= True,
            verbose = 0,
        ),
    ]

    history = keras_model.fit(
         X_tr, y_tr_s,
        validation_data = (X_val, y_val_s),
        epochs = epochs,
        batch_size = batch_size,
        callbacks = cb_list,
        verbose = 1,
    )

    _plot_training_history(history)

    y_pred = scaler_y.inverse_transform(
        keras_model.predict(X_val, verbose = 0)
    )

    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics, "ANN")
 
    keras_model.save(save_path)
    print(f"  Saved: {save_path}")

    return ModelResult(
        name = "ANN",
        model = keras_model,
        predict_fn  = _make_predict_fn(keras_model, scaler_y),
        save_path  = save_path,
        meta  = {
            "architecture": "128→64→32→4",
            "epochs_run":   len(history.history["loss"]),
            "lr": lr, "dropout": dropout, "l2": l2,
        },
    )

def load(path: str = DEFAULT_PATH, scaler_y=None) -> ModelResult:

    try: 
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow must be installed to load the ANN model.")
 
    if scaler_y is None:
        raise ValueError("scaler_y must be provided to load the ANN model.")
    
    keras_model = tf.keras.models.load_model(path)
    print(f" ANN loaded from: {path}")
 
    return ModelResult(
        name = "ANN",
        model = keras_model,
        predict_fn = _make_predict_fn(keras_model, scaler_y),
        save_path = path,
    )
