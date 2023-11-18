import streamlit as st
import numpy as np


# ? custom
from views.class_sent import Mainpage
from views.info import InfoPage


def main():
    st.set_page_config(page_title="AdSenti")
    
    st.sidebar.title("Navigation")
    # Display info in the sidebar
    page = st.sidebar.selectbox("Page", ["Info","Model"],index=0)
    # Display more intuitive info in the sidebar
    
    st.sidebar.markdown("""
    🚀 **Welcome to AdSenti!**
    
    Upload an image and receive a sentiment analysis result. 
    Choose a page from the sidebar to get started.

    **Instructions:**
    - Click on "Model" to explore the main features.
    - Navigate to "Info" for Information.
    """)

    
    # Pages
    if page == "Info":
        InfoPage()
    elif page == "Model":
        Mainpage()


if __name__ == "__main__":
    main()
