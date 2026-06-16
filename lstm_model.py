# ============================================================
#  lstm_model.py
#  Builds, trains, and evaluates one stacked LSTM per slice.
#  Predicts Ds(t+1) = [Throughput, Delay, Packet_Loss_Rate]
#
#  Key fixes vs original:
#   - PLR trained in log10 space (see preprocessing.py)
#   - inverse_transform_predictions() clips PLR >= 0
#   - predict_next_demand() returns real-scale values
# ============================================================

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import SLICES, SEQ_LEN, LSTM_UNITS, EPOCHS, BATCH_SIZE
from preprocessing import TARGET_COLS_REAL, inverse_transform_predictions

tf.random.set_seed(42)
np.random.seed(42)


def build_lstm(seq_len, n_features, n_targets, units=LSTM_UNITS):
    """
    Stacked LSTM:
      Input  : (batch, seq_len, n_features)
      Output : (batch, n_targets)  — in scaled log-space
    """
    model = Sequential([
        LSTM(units, input_shape=(seq_len, n_features),
             return_sequences=True, name="lstm_1"),
        Dropout(0.2),
        LSTM(units // 2, return_sequences=False, name="lstm_2"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(n_targets, activation="linear", name="output"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_lstm(slice_data, slice_name, epochs=EPOCHS, batch_size=BATCH_SIZE):
    Xtr, ytr = slice_data["X_train"], slice_data["y_train"]
    Xte, yte = slice_data["X_test"],  slice_data["y_test"]

    model = build_lstm(Xtr.shape[1], Xtr.shape[2], ytr.shape[1])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )
    print(f"  [{slice_name}] trained {len(history.history['loss'])} epochs  "
          f"final val_loss={history.history['val_loss'][-1]:.6f}")
    return model, history


def evaluate_lstm(model, slice_data, slice_name):
    """
    Evaluates model on test set.
    Returns y_true and y_pred in REAL scale (PLR not negative).
    """
    Xte = slice_data["X_test"]
    yte = slice_data["y_test"]
    sy  = slice_data["scaler_y"]

    y_pred_scaled = model.predict(Xte, verbose=0)
    y_pred = inverse_transform_predictions(y_pred_scaled, sy)
    y_true = inverse_transform_predictions(yte,           sy)

    metrics = {}
    for i, col in enumerate(TARGET_COLS_REAL):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"    {col:30s}  MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}")

    return y_true, y_pred, metrics


def predict_next_demand(model, slice_data):
    """
    Uses the last sequence in the test set to predict Ds(t+1).
    Returns dict {col: value} in real units, PLR clipped >= 0.
    """
    Xte  = slice_data["X_test"]
    sy   = slice_data["scaler_y"]
    last = Xte[-1:]
    pred_scaled = model.predict(last, verbose=0)
    pred_real   = inverse_transform_predictions(pred_scaled, sy)[0]
    return {
        "Throughput_Mbps":        float(max(pred_real[0], 0.0)),
        "Packet_Delay_Budget_ms": float(max(pred_real[1], 0.0)),
        "Packet_Loss_Rate":       float(np.clip(pred_real[2], 1e-9, 1.0)),
    }


def train_all_lstms(slice_data_dict):
    models, histories, metrics_all, demands, preds_plot = {}, {}, {}, {}, {}
    for slc in SLICES:
        print(f"\n── Training LSTM for {slc} ──")
        m, h = train_lstm(slice_data_dict[slc], slc)
        yt, yp, met = evaluate_lstm(m, slice_data_dict[slc], slc)
        d = predict_next_demand(m, slice_data_dict[slc])

        models[slc]      = m
        histories[slc]   = h
        metrics_all[slc] = met
        demands[slc]     = d
        preds_plot[slc]  = {"y_true": yt, "y_pred": yp}

        print(f"  [{slc}] Predicted demand → "
              f"TP={d['Throughput_Mbps']:.4f} Mbps  "
              f"Delay={d['Packet_Delay_Budget_ms']:.2f} ms  "
              f"PLR={d['Packet_Loss_Rate']:.2e}")

    return models, histories, metrics_all, demands, preds_plot




"""
Code Examples
language-python
 Copy code
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import SLICES, SEQ_LEN, LSTM_UNITS, EPOCHS, BATCH_SIZE
from preprocessing import TARGET_COLS_REAL, inverse_transform_predictions

tf.random.set_seed(42)
np.random.seed(42)
Imports: The necessary libraries are imported, including TensorFlow for building the model and scikit-learn for evaluation metrics.
Environment Variable: The TensorFlow logging level is set to suppress warnings.
language-python
 Copy code
def build_lstm(seq_len, n_features, n_targets, units=LSTM_UNITS):
    model = Sequential([
        LSTM(units, input_shape=(seq_len, n_features), return_sequences=True, name="lstm_1"),
        Dropout(0.2),
        LSTM(units // 2, return_sequences=False, name="lstm_2"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(n_targets, activation="linear", name="output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model
Function Definition: build_lstm creates a sequential model with two LSTM layers, dropout layers for regularization, and dense layers for output.
Compilation: The model is compiled with Adam optimizer and mean squared error loss.
language-python
 Copy code
def train_lstm(slice_data, slice_name, epochs=EPOCHS, batch_size=BATCH_SIZE):
    Xtr, ytr = slice_data["X_train"], slice_data["y_train"]
    Xte, yte = slice_data["X_test"], slice_data["y_test"]

    model = build_lstm(Xtr.shape[1], Xtr.shape[2], ytr.shape[1])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0, shuffle=False)
    print(f"  [{slice_name}] trained {len(history.history['loss'])} epochs  final val_loss={history.history['val_loss'][-1]:.6f}")
    return model, history
Training Function: train_lstm trains the model using training and validation data, applying early stopping and learning rate reduction callbacks.
Model Fitting: The model is fitted to the training data, and the training history is returned.
language-python
 Copy code
def evaluate_lstm(model, slice_data, slice_name):
    Xte = slice_data["X_test"]
    yte = slice_data["y_test"]
    sy  = slice_data["scaler_y"]

    y_pred_scaled = model.predict(Xte, verbose=0)
    y_pred = inverse_transform_predictions(y_pred_scaled, sy)
    y_true = inverse_transform_predictions(yte, sy)

    metrics = {}
    for i, col in enumerate(TARGET_COLS_REAL):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"    {col:30s}  MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}")

    return y_true, y_pred, metrics
Evaluation Function: evaluate_lstm assesses the model's performance on the test set, calculating various metrics for each target column.
Inverse Transformation: Predictions and true values are transformed back to their original scale for accurate evaluation.
language-python
 Copy code
def predict_next_demand(model, slice_data):
    Xte  = slice_data["X_test"]
    sy   = slice_data["scaler_y"]
    last = Xte[-1:]
    pred_scaled = model.predict(last, verbose=0)
    pred_real   = inverse_transform_predictions(pred_scaled, sy)[0]
    return {
        "Throughput_Mbps":        float(max(pred_real[0], 0.0)),
        "Packet_Delay_Budget_ms": float(max(pred_real[1], 0.0)),
        "Packet_Loss_Rate":       float(np.clip(pred_real[2], 1e-9, 1.0)),
    }
Demand Prediction: predict_next_demand uses the last sequence of the test data to predict future values, ensuring that packet loss rate is non-negative.
language-python
 Copy code
def train_all_lstms(slice_data_dict):
    models, histories, metrics_all, demands, preds_plot = {}, {}, {}, {}, {}
    for slc in SLICES:
        print(f"\n── Training LSTM for {slc} ──")
        m, h = train_lstm(slice_data_dict[slc], slc)
        yt, yp, met = evaluate_lstm(m, slice_data_dict[slc], slc)
        d = predict_next_demand(m, slice_data_dict[slc])

        models[slc]      = m
        histories[slc]   = h
        metrics_all[slc] = met
        demands[slc]     = d
        preds_plot[slc]  = {"y_true": yt, "y_pred": yp}

        print(f"  [{slc}] Predicted demand → "
              f"TP={d['Throughput_Mbps']:.4f} Mbps  "
              f"Delay={d['Packet_Delay_Budget_ms']:.2f} ms  "
              f"PLR={d['Packet_Loss_Rate']:.2e}")

    return models, histories, metrics_all, demands, preds_plot
Training All Models: train_all_lstms iterates through multiple data slices, training and evaluating an LSTM for each, and storing the results for further analysis.
Conclusion
The provided script effectively demonstrates how to build, train, and evaluate a stacked LSTM model for time series prediction. By leveraging TensorFlow and Keras, it incorporates essential practices such as early stopping and performance metrics evaluation. This approach can be adapted for various applications in predictive analytics, particularly in network performance forecasting.

"""