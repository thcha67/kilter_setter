import tensorflow as tf
import os, sys
sys.path.append(os.getcwd())
from src.neural_network.keras_gan import get_maks
import matplotlib.pyplot as plt

mask = get_maks()

test = mask* 12

def custom_loss_with_mask():
    def loss(y_true, y_pred):
        # Penalize for wrong values
        value_mask = tf.cast(tf.math.logical_or(
            tf.math.logical_or(y_pred == 0, y_pred == 12),
            tf.math.logical_or(y_pred == 13, 
                tf.math.logical_or(y_pred == 14, y_pred == 15))), 
            tf.float32)


        # Penalize for wrong positions
        position_mask = tf.cast(get_maks(), tf.float32)

        combined_mask = value_mask * position_mask

        # Calculate loss
        print('y_true:', y_true)
        plt.imshow(y_true)
        plt.show()
        print('y_pred:', y_pred)
        plt.imshow(y_pred)
        plt.show()
        print('y_true * combined_mask:', y_true * combined_mask)
        plt.imshow(y_true * combined_mask)
        plt.show()
        print('y_pred * combined_mask:', y_pred * combined_mask)
        plt.imshow(y_pred * combined_mask)
        plt.show()
        calculated_loss = tf.keras.losses.mean_squared_error(y_true, y_pred * combined_mask)
        print("Calculated Loss:", calculated_loss.numpy())
        return calculated_loss
    return loss

loss = custom_loss_with_mask()

mask += 1
print(loss(test, mask))