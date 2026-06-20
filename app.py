import streamlit as st
import joblib
import os
import numpy as np
import autoai_libs
import xgboost
import snapml

# Page Configuration
st.set_page_config(page_title="AI News Verifier", page_icon="📰", layout="centered")

# Secure Model Loader
@st.cache_resource
def load_watson_model():
    model_path = "welfake_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model file: {e}")
            return None
    return None

model = load_watson_model()

# UI Layout
st.title("📰 AI-Powered Fake News Detector")
st.markdown("---")
st.write("Provide the headline and article text details below to analyze authenticity.")

news_title = st.text_input("Article Title", placeholder="Enter the news headline...")
news_body = st.text_area("Article Content", placeholder="Paste the complete text body of the article here...", height=200)

# Prediction Logic
if st.button("Verify Authenticity", type="primary"):
    if not news_title.strip() or not news_body.strip():
        st.warning("⚠️ Please provide both the title and text content.")
    else:
        with st.spinner("Analyzing linguistic patterns..."):
            if model is not None:
                try:
                    # Verified 2D matrix structure required by the IBM AutoAI Pipeline
                    input_matrix = np.array([[0, news_title, news_body]], dtype=object)
                    prediction = model.predict(input_matrix)[0]
                    
                    st.markdown("### 📊 Classification Output")
                    pred_val = str(prediction).strip().upper()
                    
                    if pred_val in ['1', '1.0', 'REAL', 'TRUE']:
                        st.success("✅ **Prediction: REAL NEWS**")
                    else:
                        st.error("🚨 **Prediction: FAKE NEWS**")
                        
                    st.info("Analysis executed smoothly via your exported IBM AutoAI Pipeline.")
                except Exception as e:
                    st.error("❌ **Pipeline Processing Error**")
                    st.code(str(e))
            else:
                st.error("Model file 'welfake_model.pkl' could not be found.")
