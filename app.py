import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = r"C:\Users\HP\Documents\sfgr\archive\signatures\xgboost_model.pkl"


# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        font-size: 36px;
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
    }
    .subtitle {
        font-size: 20px;
        color: #7f8c8d;
        text-align: center;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .result-box {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Signature Predictor Class (from optimized code)
class SignaturePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")

    def preprocess_image(self, image):
        try:
            # Convert PIL Image to OpenCV format
            image = np.array(image.convert('L'))  # Grayscale
            image = cv2.resize(image, (128, 128))
            features = hog(
                image,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                visualize=False
            )
            return features
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, image):
        hog_features = self.preprocess_image(image)
        if hog_features is None:
            return "Invalid Image"
        hog_features = np.array(hog_features).reshape(1, -1)
        prediction = self.model.predict(hog_features)
        return "Forged" if prediction[0] == 1 else "Genuine"

# Initialize predictor
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure 'xgboost_model.pkl' is in the current directory.")
    st.stop()
predictor = SignaturePredictor(MODEL_PATH)

# App Layout
def main():
    st.title("Signature Fraud Detection", anchor="title")
    st.markdown("<p class='subtitle'>Upload a signature image to verify its authenticity</p>", unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Performance Metrics", "About"])

    if page == "Home":
        # Image Upload Section
        st.subheader("Upload Signature Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Signature", use_container_width=True)

            # Predict Button
            if st.button("Verify Signature"):
                with st.spinner("Analyzing..."):
                    result = predictor.predict(image)
                # Display result with styling
                if result == "Genuine":
                    st.markdown(f"<div class='result-box' style='color: #27ae60;'>Result: {result}</div>", unsafe_allow_html=True)
                elif result == "Forged":
                    st.markdown(f"<div class='result-box' style='color: #c0392b;'>Result: {result}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-box' style='color: #7f8c8d;'>Result: {result}</div>", unsafe_allow_html=True)

    elif page == "Performance Metrics":
        st.subheader("Model Performance Metrics")
        # Placeholder metrics (replace with actual values from your training)
        metrics = {
            "Accuracy": "92.50%",
            "Precision (Genuine)": "94.00%",
            "Recall (Genuine)": "91.00%",
            "F1-Score (Genuine)": "92.50%",
            "Precision (Forged)": "91.00%",
            "Recall (Forged)": "93.00%",
            "F1-Score (Forged)": "92.00%"
        }
        cols = st.columns(4)
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % 4]:
                st.markdown(f"<div class='metric-box'><b>{metric}</b><br>{value}</div>", unsafe_allow_html=True)

    elif page == "About":
        st.subheader("About This Project")
        st.write("""
            This Signature Fraud Detection system uses advanced machine learning techniques to classify signatures as genuine or forged. 
            It leverages Histogram of Oriented Gradients (HOG) features and an XGBoost classifier, trained on a dataset of signature images.
            
            *Features:*
            - Upload signature images for real-time verification
            - View model performance metrics
            - Responsive and user-friendly interface
            
            Developed by Team RGUKT Winners using Streamlit and Python under the guidance of Srikanth sir.
        """)

if __name__ == "__main__":
    main()