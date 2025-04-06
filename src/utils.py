import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import sys

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for model prediction.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (height, width)
    
    Returns:
        numpy.ndarray: Preprocessed image array
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded or processed
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and resize image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        
        # Expand dimensions and normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def visualize_prediction(image_path, prediction, confidence):
    """
    Visualize the model's prediction on an image.
    
    Args:
        image_path (str): Path to the input image
        prediction (int): Binary prediction (0 or 1)
        confidence (float): Prediction confidence score
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded or processed
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and display original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        
        # Add prediction text
        result = "Normal" if prediction == 0 else "Pneumonia"
        plt.title(f"Prediction: {result}\nConfidence: {confidence:.2%}")
        plt.axis('off')
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save the visualization
        plt.savefig('results/last_prediction.png')
        plt.close()
    except Exception as e:
        print(f"Error visualizing prediction: {str(e)}")
        raise

def create_confusion_matrix_plot(cm):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        cm (numpy.ndarray): 2x2 confusion matrix
    
    Raises:
        ValueError: If the confusion matrix is not 2x2
    """
    try:
        if cm.shape != (2, 2):
            raise ValueError("Confusion matrix must be 2x2")

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        # Add labels
        classes = ['Normal', 'Pneumonia']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save the plot
        plt.savefig('results/confusion_matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {str(e)}")
        raise 