import streamlit as st

def InfoPage():
    # Title
    st.title("Image Sentiment Analysis App")

    # Information section
    st.markdown("""
    Welcome to the Image Sentiment Analysis app! This tool allows you to upload an image and receive a sentiment analysis result. 
    The analysis is based on a pre-trained model or a sentiment analysis service.
    """)

    # How to Use section
    st.markdown("### How to Use")

    st.markdown("""
    1. **Upload an Image:**
       - Click on the "Choose an image..." button.
       - Select an image file in JPG, PNG, or JPEG format.

    2. **View Uploaded Image:**
       - The uploaded image will be displayed on the app.

    3. **Sentiment Analysis:**
       - The app will perform sentiment analysis on the uploaded image.
       - The sentiment result will be displayed as either positive, negative, or neutral.
    """)

    # Important Notes section
    st.markdown("### Important Notes")

    st.markdown("""
    - Ensure that the uploaded image is clear and relevant to sentiment analysis.
    - The accuracy of the sentiment analysis depends on the underlying model or service.
    """)

    # About the App section
    st.markdown("### About the App")

    st.markdown("""
    This app is built using Streamlit, a simple and powerful framework for creating web applications with Python. 
    The sentiment analysis functionality is powered by CNN classification.
    """)

    # Disclaimer section
    st.markdown("### Disclaimer")

    st.markdown("""
    This tool is for demonstration purposes only. The accuracy of sentiment analysis may vary, and the results should not be considered absolute.
    Feel free to explore the app and analyze different images for their sentiment!
    For any questions or issues, please contact +918828388979.
    """)
