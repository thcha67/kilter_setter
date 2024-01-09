import numpy as np
import sqlite3
import sys, os
sys.path.append(os.getcwd())
from tensorflow.keras import models, layers
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

# model = create_model()

mask = get_maks()

def custom_loss(y_true, y_pred):
    mask = tf.cast(tf.math.logical_or(
        tf.math.logical_or(y_pred == 0, y_pred == 12),
        tf.math.logical_or(y_pred == 13, 
            tf.math.logical_or(y_pred == 14, y_pred == 15))), 
        tf.float32)
    return tf.keras.losses.mean_squared_error(y_true * mask, y_pred * mask)



