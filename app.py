import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def preprocess_image(image):
    image = tf.image.decode_image(image.read(), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    return image

# Load the pre-trained model
model = tf.keras.models.load_model('lungcaner_detection2.h5')

# Streamlit UI
st.title("Lung Cancer Detection")

# Upload an image
st.subheader("Upload an image for analysis")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = preprocess_image(uploaded_image)
    # Convert TensorFlow tensor to NumPy array
    image = image.numpy()
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    if st.button("Detect Lung Cancer"):
        st.write("Detecting lung cancer...")

        # Make predictions using the model
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, axis=0)
        predictions = model.predict(image)

         # Get the top predicted class and confidence
        class_id = np.argmax(predictions.round())
        confidence = predictions[class_id]

        st.write(f"Predicted Class ID: {class_id}")
        st.write(f"Confidence: {confidence:.4f}")

# # Optional: Display a sample image for testing
# sample_image = Image.open("sample_lung_image.jpg")
# st.image(sample_image, caption="Sample Lung Image", use_column_width=True)
