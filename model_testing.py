import tensorflow as tf
import numpy as np

def preprocess_image(image):
    image = tf.image.decode_image(image.read(), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (200,200))
    image = image.numpy()
    return image


model = tf.keras.models.load_model('lungcaner_detection2.h5')

inp_image = preprocess_image("NORMAL2-IM-0281-0001.jpeg")

model.predict(inp_image)