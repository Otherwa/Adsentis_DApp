from pymongo import MongoClient
from bson import ObjectId
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import base64
import os
from dotenv import load_dotenv
from streamlit_extras.let_it_rain import rain

load_dotenv()

# Define the label with which you want to classify
# custom_class_labels = [
#     "alert",
#     "afraid",
#     "angry",
#     "amused",
#     "calm",
#     "alarmed",
#     "amazed",
#     "cheerful",
#     "active",
#     "conscious",
#     "creative",
#     "educative",
#     "grateful",
#     "confident",
#     "disturbed",
#     "emotional",
#     "fashionable",
#     "empathetic",
#     "feminine",
#     "eager",
#     "inspired",
#     "jealous",
#     "proud",
#     "pessimistic",
#     "manly",
#     "sad",
#     "persuaded",
#     "loving",
#     "youthful",
#     "thrifty",
# ]

custom_class_labels = ["active", "afraid", "alaramed", "amazed ", "angry"]


@st.cache_data()
def InitModel(model_path):
    with st.spinner("Wait for it..."):
        st.write("Model Loaded ⭐")
        return load_model(model_path)


@st.cache_data()
def preprocess_image(img):
    with st.spinner("Wait for it..."):
        img_array = image.img_to_array(img)
        st.write("Image Normalized.")
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0


def predict_image(model, img_array):
    with st.spinner("* Wait for it..."):
        return model.predict(img_array)


def display_results(img, predicted_class_label, confidence):
    with st.status("* Predictions", expanded=True):
        st.image(img, caption="Uploaded Image", width=500)
        st.subheader("Prediction:")
        st.info(f"Predicted class label: {predicted_class_label}")
        st.warning(f"Prediction Confidence: {confidence:.2%}")


def save_to_mongodb(img_base64, predicted_class_label, confidence):
    # Connect to MongoDB

    # Add a button for user interaction
    if st.button("Was the response successful? Click 'Yes' to confirm."):
        # Save to MongoDB
        save_to_mongodb_impl(img_base64, predicted_class_label, confidence)


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
                "confidence": confidence,
            }
        )

        rain(
            emoji="👍",
            font_size=54,
            falling_speed=9,
            animation_length="500",
        )
        st.success("* Your Response Was Successfully Recorded")


def main():
    st.set_page_config(page_title="🤡 AdSenti", layout="wide")
    st.header("🤡 AdSent")
    st.title("AdSenti (Image Sentiment Analysis)")
    st.warning(
        f"* Only 5 Emotions are classified as per now 'active', 'afraid', 'alaramed', 'amazed ', 'angry'"
    )
    st.error("* Note As per our Dataset")
    st.info("* Help us by testing your side of images for classifiaction")
    model_path = "./config/model/Ads_Senti_128bs_35ep.keras"
    model = InitModel(model_path)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = preprocess_image(img)

        predictions = predict_image(model, img_array)

        predicted_class_index = np.argmax(predictions)
        predicted_class_label = custom_class_labels[predicted_class_index]

        # Create a status container for dynamic updates
        status_container = st.empty()

        # Display results and save to MongoDB
        display_results(
            img, predicted_class_label, predictions[0][predicted_class_index]
        )
        img_base64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        save_to_mongodb(
            img_base64,
            predicted_class_label,
            str(predictions[0][predicted_class_index]),
        )


if __name__ == "__main__":
    main()
