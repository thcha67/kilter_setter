import numpy as np
import sqlite3
import sys, os
sys.path.append(os.getcwd())
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from scripts.kilter_utils import get_matrix_from_holes, get_all_holes_12x12
import matplotlib.pyplot as plt


def get_maks():
    all_holes = get_all_holes_12x12()
    all_holes = get_matrix_from_holes(all_holes)
    # Make 1 if value is not zero, 0 otherwise
    all_holes = np.where(all_holes == 0, 0, 1)
    return all_holes
    

def normalize_matrix(matrix):
    return matrix / np.max(matrix)

def load_training_data(path):
    training_data = np.load(path).astype(np.float32)
    print(training_data.shape)
    return training_data

inputs, targets = load_training_data('data/inputs.npy'), load_training_data('data/targets.npy')

train_inputs = inputs[:len(inputs) // 10 *8]
test_inputs = inputs[len(inputs) // 10 *8:]

train_targets = targets[:len(targets) // 10 *8]
test_targets = targets[len(targets) // 10 *8:]


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(2,)))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(157 * 145, activation='linear'))  # Output layer
    model.add(layers.Reshape((157, 145)))
    return model

model = create_model()

allowed_positions_mask = get_maks()
print(allowed_positions_mask.shape)

def custom_loss_with_mask(allowed_positions_mask):
    def loss(y_true, y_pred):
        # Penalize for wrong values
        value_mask = tf.cast(tf.math.logical_or(
            tf.math.logical_or(y_pred == 0, y_pred == 12),
            tf.math.logical_or(y_pred == 13, 
                tf.math.logical_or(y_pred == 14, y_pred == 15))), 
            tf.float32)

        # Penalize for wrong positions
        position_mask = tf.cast(allowed_positions_mask, tf.float32)
        combined_mask = value_mask * position_mask

        # Calculate loss
        return tf.keras.losses.mean_squared_error(y_true * combined_mask, y_pred * combined_mask)

    return loss

model.compile(optimizer='adam', loss=custom_loss_with_mask(allowed_positions_mask))

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# history = model.fit(train_inputs, train_targets, 
#                     batch_size=32, 
#                     epochs=100, 
#                     validation_data=(test_inputs, test_targets),
#                     callbacks=[early_stopping])

