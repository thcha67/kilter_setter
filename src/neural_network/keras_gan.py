import numpy as np
import sqlite3
import sys, os
sys.path.append(os.getcwd())
import tensorflow as tf
from scripts.kilter_utils import get_matrix_from_holes, get_all_holes_12x12
from tensorflow.keras import models, layers


def get_maks():
    all_holes = get_all_holes_12x12()
    all_holes = get_matrix_from_holes(all_holes)
    return all_holes
    

def normalize_matrix(matrix):
    return matrix / matrix.max()

def load_training_data(path):
    training_data = np.load(path).astype(np.float32)
    inputs = training_data[:, :2]  # (num_samples, 2)
    targets = training_data[:, 2:]  #  (num_samples, 157*161)
    targets = normalize_matrix(targets)
    return inputs, targets

inputs, targets = load_training_data('data/inputs.npy'), load_training_data('data/targets.npy')

train_inputs = inputs[:len(inputs) // 10 *8]
test_inputs = inputs[len(inputs) // 10 *8:]

train_targets = targets[:len(targets) // 10 *8]
test_targets = targets[len(targets) // 10 *8:]


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(2,)))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(157 * 161, activation='linear'))  # Output layer
    model.add(layers.Reshape((157, 161)))
    return model

model = create_model()

