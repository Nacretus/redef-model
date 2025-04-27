import tensorflow as tf
from tensorflow import keras

print (tf.__version__)

print (tf.keras.__version__)


model = tf.keras.models.load_model('best_model_Nadam.keras')