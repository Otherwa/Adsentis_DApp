'''
? Main Page fuctions
'''

import base64
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
import tensorflow as tf
from dotenv import load_dotenv
import random
import os
import numpy as np
from streamlit_extras.let_it_rain import rain

load_dotenv()

# Define the label with which you want to classify
custom_class_labels = [
    "alert",
    "afraid",
    "angry",
    "amused",
    "calm",
    "alarmed",
    "amazed",
    "cheerful",
    "active",
    "conscious",
    "creative",
    "educative",
    "grateful",
    "confident",
    "disturbed",
    "emotional",
    "fashionable",
    "empathetic",
    "feminine",
    "eager",
    "inspired",
    "jealous",
    "proud",
    "pessimistic",
    "manly",
    "sad",
    "persuaded",
    "loving",
    "youthful",
    "thrifty",
]


@st.cache_data()
def InitModel(model_path):
    with st.spinner("Wait for it..."):
        st.write("Model Loaded ⭐")
        return load_model(model_path)


@st.cache_data()
def preprocess_image(img):
    with st.spinner("Wait for it..."):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0


# ! Call back issue
@tf.function(reduce_retracing=True)
def predict_image(model, img_array):
    with st.spinner("* Wait for it..."):
        return model(img_array)


def display_results(img, predictions, custom_class_labels):
    with st.status("* Predictions", expanded=False) as status:
        st.image(
            img,
            caption="Uploaded Image",
            output_format="PNG",
            use_column_width="always",
        )
        st.subheader("Top Predictions:")

        # Combine and zip the two
        zipped_data = list(zip(predictions, custom_class_labels))
        # Sort by the probabilities in descending order
        sorted_data = sorted(zipped_data, key=lambda x: x[0], reverse=True)
        emojis = ['😊', '🌟', '🎉', '🐍', '🚀', '💻', '🤖', '🌈', '🍕', '🎸','👾','✅','📨','⭐']
        for pred, label in sorted_data[:6]:
            st.info(f"{label.title()} : {pred:.2%}", icon=random.choice(emojis))
        status.update(label="Predictions complete!", state="complete", expanded=True)


def save_to_mongodb(img_base64, predicted_class_label, confidence):
    # Connect to MongoDB

    # Add a button for user interaction
    if st.button("Was the response successful? Click to confirm."):
        # Save to MongoDB
        save_to_mongodb_impl(img_base64, predicted_class_label, confidence)
        # Update session state to indicate that the button has been clicked


def save_to_mongodb_impl(img_base64, predicted_class_label, confidence):
    # Connect to MongoDB
    st.info("* Saving to MongoDB...")
    with st.spinner("Wait for it..."):
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("DB_NAME")]
        collection = db["image_predictions"]

        # Store the image and prediction information in MongoDB
        collection.insert_one(
            {
                "image": img_base64,
                "predicted_class_label": predicted_class_label,
                "confidence": float(confidence),
            }
        )

        rain(
            emoji="👍",
            font_size=54,
            falling_speed=9,
            animation_length="500",
        )
        st.success("* Your Response Was Successfully Recorded")

def Mainpage():
 
    st.header("Image Sentiment Analysis")
    st.warning("* Note As per our Dataset")
    st.info("* Help us by testing your side of images for classifiaction")
    # ! Model 
    MODEL_PATH = "./config/model/Ads_Senti_Real_128bs_35ep.keras"
    
    model = InitModel(MODEL_PATH)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        show_img = image.load_img(uploaded_file)
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = preprocess_image(img)
        predictions = predict_image(model, img_array)
        # display results
        display_results(show_img, predictions[0], custom_class_labels)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = custom_class_labels[predicted_class_index]
        # Extracting the numerical value
        tensor_value = predictions[0][predicted_class_index]
        numeric_value = tensor_value.numpy()
        img_base64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        save_to_mongodb(
            img_base64,
            predicted_class_label,
            str(numeric_value),
        )