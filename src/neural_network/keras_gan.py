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



def create_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(2,)))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(157 * 145, activation='linear'))  # Output layer
    model.add(layers.Reshape((157, 145)))
    return model

# def create_model():
#     model = tf.keras.Sequential()

#     # Dense layer and reshape
#     model.add(layers.Dense(38*38*256, use_bias=False, input_shape=(2,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Reshape((38, 38, 256)))

#     # Conv2DTranspose layers
#     model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())

#     model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

#     # Precise cropping
#     crop_height_top = (model.output_shape[1] - 157) // 2
#     crop_height_bottom = model.output_shape[1] - 157 - crop_height_top
#     crop_width_left = (model.output_shape[2] - 145) // 2
#     crop_width_right = model.output_shape[2] - 145 - crop_width_left

#     model.add(layers.Cropping2D(cropping=((crop_height_top, crop_height_bottom), (crop_width_left, crop_width_right))))

#     return model



def custom_loss_with_mask(allowed_positions_mask):
    def loss(y_true, y_pred):
        # Penalize for wrong values
        value_mask = tf.cast(tf.math.logical_or(y_pred < 1, y_pred > 11.5), tf.float32)

        # Penalize for wrong positions
        position_mask = tf.cast(allowed_positions_mask, tf.float32)
        combined_mask = value_mask * position_mask

        # Calculate loss
        return tf.keras.losses.mean_squared_error(y_true, y_pred * combined_mask)

    return loss

if __name__ == '__main__':
    
    inputs, targets = load_training_data('data/inputs.npy'), load_training_data('data/targets.npy')

    train_inputs = inputs[:len(inputs) // 10 *8]
    test_inputs = inputs[len(inputs) // 10 *8:]

    train_targets = targets[:len(targets) // 10 *8]
    test_targets = targets[len(targets) // 10 *8:]
    model = create_model()

    allowed_positions_mask = get_maks()
    print(allowed_positions_mask.shape)
    model.compile(optimizer='adam', loss=custom_loss_with_mask(allowed_positions_mask))

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(train_inputs, train_targets, 
                        batch_size=16, 
                        epochs=100, 
                        validation_data=(test_inputs, test_targets),
                        callbacks=[early_stopping])

    test_loss = model.evaluate(test_inputs, test_targets)
    print("Test Loss:", test_loss)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

