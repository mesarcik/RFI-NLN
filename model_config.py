#config for training
import tensorflow as tf

BUFFER_SIZE =25000 # 60000
BATCH_SIZE = 2**10
bce = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
mae=tf.keras.losses.MeanAbsoluteError()

### AE Parameters for CIFAR, MNIST, FMNIST 
n_filters =32 
n_layers =2
