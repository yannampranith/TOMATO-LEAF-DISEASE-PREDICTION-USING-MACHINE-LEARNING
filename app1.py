import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Load your trained model
MODEL_PATH ='model_inception.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = tf.image.resize(x, (224, 224))  # Resize for the model
    x = tf.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = tf.argmax(preds, axis=1).numpy()[0]

    class_labels = [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
        "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "Healthy"
    ]
    result = class_labels[preds]
    
    return result

def main():
    st.title("Plant Disease Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make prediction
        preds = model_predict(uploaded_file, model)
        st.success(f"Prediction: {preds}")

if __name__ == '__main__':
    main()
