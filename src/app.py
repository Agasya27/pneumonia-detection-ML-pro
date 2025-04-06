import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from model import get_model
import io
import sys

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .confidence-bar {
        height: 25px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 5px 0;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        background-color: #1E88E5;
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
    .confidence-text {
        position: absolute;
        width: 100%;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 25px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">Pneumonia Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Upload a chest X-ray image to detect if the patient has pneumonia.</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">About</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="info-text">
    This application uses a deep learning model to analyze chest X-ray images and detect signs of pneumonia.
    
    The model was trained on a dataset of chest X-ray images and can classify images as either normal or showing signs of pneumonia.
    
    Upload an X-ray image to get an instant prediction.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="info-text">
    - Model: Custom CNN Architecture<br>
    - Input Size: 160x160 pixels<br>
    - Classes: Normal, Pneumonia<br>
    - Training: 5,216 images<br>
    - Validation: 10 images
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Threshold Adjustment</h2>', unsafe_allow_html=True)
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05, 
                         help="Adjust the threshold for classification. Higher values make the model more conservative in predicting pneumonia.")
    
    # Add a section for validation images
    st.markdown('<h2 class="sub-header">Validation Images</h2>', unsafe_allow_html=True)
    validation_option = st.radio(
        "Select validation image to test:",
        ["None", "Normal", "Pneumonia"]
    )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="sub-header">Upload X-ray Image</h2>', unsafe_allow_html=True)
    
    # Handle validation image selection
    if validation_option != "None":
        if validation_option == "Normal":
            val_dir = "data/val/NORMAL"
            val_files = [f for f in os.listdir(val_dir) if f.endswith('.jpeg')]
            if val_files:
                selected_file = st.selectbox("Select a normal validation image:", val_files)
                image_path = os.path.join(val_dir, selected_file)
                image = Image.open(image_path).convert('RGB')
                st.image(image, caption=f"Selected: {selected_file}", use_container_width=True)
            else:
                st.warning("No validation images found in the NORMAL directory.")
                image = None
        else:  # Pneumonia
            val_dir = "data/val/PNEUMONIA"
            val_files = [f for f in os.listdir(val_dir) if f.endswith('.jpeg')]
            if val_files:
                selected_file = st.selectbox("Select a pneumonia validation image:", val_files)
                image_path = os.path.join(val_dir, selected_file)
                image = Image.open(image_path).convert('RGB')
                st.image(image, caption=f"Selected: {selected_file}", use_container_width=True)
            else:
                st.warning("No validation images found in the PNEUMONIA directory.")
                image = None
    else:
        # Regular file upload
        uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
        else:
            image = None
    
    # Add a predict button
    if image is not None:
        if st.button("Analyze Image"):
            try:
                with st.spinner("Analyzing image..."):
                    # Load model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = get_model(device)
                    
                    # Check if model exists
                    model_path = Path('models/pneumonia_model.pth')
                    if not model_path.exists():
                        st.error("Model file not found. Please train the model first.")
                    else:
                        # Load model weights
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        model.eval()
                        
                        # Preprocess image
                        transform = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                        ])
                        
                        img_tensor = transform(image).unsqueeze(0).to(device)
                        
                        # Get prediction
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            probabilities = torch.softmax(outputs, dim=1)[0]
                            normal_prob = probabilities[0].item()
                            pneumonia_prob = probabilities[1].item()
                            
                            # Apply threshold
                            prediction = "PNEUMONIA" if pneumonia_prob > threshold else "NORMAL"
                            confidence = pneumonia_prob if prediction == "PNEUMONIA" else normal_prob
                            
                            # Display results in the second column
                            with col2:
                                st.markdown('<h2 class="sub-header">Analysis Results</h2>', unsafe_allow_html=True)
                                
                                # Result box
                                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                
                                # Prediction
                                result_color = "#FF5252" if prediction == "PNEUMONIA" else "#4CAF50"
                                st.markdown(f'<h3 style="color: {result_color};">Prediction: {prediction}</h3>', unsafe_allow_html=True)
                                
                                # Confidence
                                st.markdown(f'<p>Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)
                                
                                # Confidence bars
                                st.markdown('<p>NORMAL:</p>', unsafe_allow_html=True)
                                st.markdown(f'<div class="confidence-bar"><div class="confidence-fill" style="width: {normal_prob:.0%}"></div><div class="confidence-text">{normal_prob:.1%}</div></div>', unsafe_allow_html=True)
                                
                                st.markdown('<p>PNEUMONIA:</p>', unsafe_allow_html=True)
                                st.markdown(f'<div class="confidence-bar"><div class="confidence-fill" style="width: {pneumonia_prob:.0%}"></div><div class="confidence-text">{pneumonia_prob:.1%}</div></div>', unsafe_allow_html=True)
                                
                                # Ground truth for validation images
                                if validation_option != "None":
                                    ground_truth = "NORMAL" if validation_option == "Normal" else "PNEUMONIA"
                                    correct = prediction == ground_truth
                                    result_text = "✅ Correct" if correct else "❌ Incorrect"
                                    result_color = "#4CAF50" if correct else "#FF5252"
                                    st.markdown(f'<h3 style="color: {result_color};">Ground Truth: {ground_truth} - {result_text}</h3>', unsafe_allow_html=True)
                                
                                # Recommendations
                                st.markdown('<h3>Recommendations:</h3>', unsafe_allow_html=True)
                                if prediction == "PNEUMONIA":
                                    st.markdown("""
                                    - Consult a healthcare provider for further evaluation
                                    - Consider getting a chest CT scan for more detailed imaging
                                    - Monitor symptoms and seek immediate medical attention if breathing difficulties worsen
                                    """)
                                else:
                                    st.markdown("""
                                    - No signs of pneumonia detected in this X-ray
                                    - Continue regular health check-ups
                                    - If experiencing symptoms, consult a healthcare provider
                                    """)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Disclaimer
                                st.markdown("""
                                <p style="font-size: 0.8rem; color: #9E9E9E; margin-top: 20px;">
                                <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.
                                </p>
                                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please make sure the model is properly trained and all dependencies are installed correctly.")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p style="color: #616161;">© 2025 Pneumonia Detection System | Developed with ❤️ for healthcare</p>
</div>
""", unsafe_allow_html=True) 