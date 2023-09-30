import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def preprocess_image(image):
    image = tf.image.decode_image(image.read(), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (200, 200))
    return image

# Load the pre-trained model
model = tf.keras.models.load_model('lung_cancer_simple.h5')

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
        image = tf.image.resize(image, (200,200))
        image = tf.expand_dims(image, axis=0)
        predictions = model.predict(image)

        

        # Assuming you're doing binary classification, you can round the prediction
        rounded_predictions = np.round(predictions)


        class_id = rounded_predictions[0][0]
        confidence = predictions[0][0]

        class_labels = ["Negative", "Positive"]  

        st.write(f"Predicted Class ID: {class_labels[int(class_id)]}")
        st.write(f"Confidence: {confidence:.4f}")
