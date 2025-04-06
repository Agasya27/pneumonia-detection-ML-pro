import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import get_model

class PneumoniaPredictor:
    def __init__(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = get_model(self.device)
        self.model.load_state_dict(torch.load('models/pneumonia_model.pth', map_location=self.device))
        self.model.eval()
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Pneumonia Detection System")
        self.root.geometry("900x700")
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Create image display
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(pady=10)
        
        # Create threshold adjustment
        self.threshold_frame = tk.Frame(self.main_frame)
        self.threshold_frame.pack(pady=10)
        
        self.threshold_label = tk.Label(self.threshold_frame, text="Prediction Threshold:", font=("Arial", 12))
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_value = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(self.threshold_frame, from_=0.0, to=1.0, 
                                        orient=tk.HORIZONTAL, variable=self.threshold_value,
                                        command=self.update_prediction)
        self.threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.threshold_display = tk.Label(self.threshold_frame, text="0.50", font=("Arial", 12))
        self.threshold_display.pack(side=tk.LEFT, padx=5)
        
        # Create result display
        self.result_frame = tk.Frame(self.main_frame)
        self.result_frame.pack(pady=10)
        
        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"))
        self.result_label.pack()
        
        # Create confidence display
        self.confidence_label = tk.Label(self.result_frame, text="", font=("Arial", 12))
        self.confidence_label.pack()
        
        # Create probability bars
        self.prob_frame = tk.Frame(self.main_frame)
        self.prob_frame.pack(pady=10)
        
        self.normal_prob = tk.Label(self.prob_frame, text="Normal: 0%", font=("Arial", 12))
        self.normal_prob.pack()
        
        self.pneumonia_prob = tk.Label(self.prob_frame, text="Pneumonia: 0%", font=("Arial", 12))
        self.pneumonia_prob.pack()
        
        # Create buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=20)
        
        self.select_button = tk.Button(self.button_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_image)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize variables
        self.current_image_path = None
        self.current_image = None
        self.current_probabilities = None
        
    def select_image(self):
        """Open file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            title="Select Chest X-ray Image",
            filetypes=[("Image files", "*.jpeg *.jpg *.png")]
        )
        
        if file_path:
            self.current_image_path = file_path
            # Display the image
            self.display_image(file_path)
            # Clear previous results
            self.clear_results()
    
    def display_image(self, image_path):
        """Display the selected image in the GUI."""
        # Open and resize image
        image = Image.open(image_path)
        # Calculate resize ratio to fit in 400x400 while maintaining aspect ratio
        ratio = min(400/image.width, 400/image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Update image display
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def clear_results(self):
        """Clear all result displays."""
        self.result_label.config(text="")
        self.confidence_label.config(text="")
        self.normal_prob.config(text="Normal: 0%")
        self.pneumonia_prob.config(text="Pneumonia: 0%")
        self.current_probabilities = None
    
    def update_prediction(self, *args):
        """Update prediction based on current threshold."""
        if self.current_probabilities is None:
            return
        
        threshold = self.threshold_value.get()
        self.threshold_display.config(text=f"{threshold:.2f}")
        
        normal_prob, pneumonia_prob = self.current_probabilities
        
        # Update probability displays
        self.normal_prob.config(text=f"Normal: {normal_prob:.1%}")
        self.pneumonia_prob.config(text=f"Pneumonia: {pneumonia_prob:.1%}")
        
        # Determine prediction based on threshold
        if pneumonia_prob >= threshold:
            result = "PNEUMONIA"
            color = "red"
        else:
            result = "NORMAL"
            color = "green"
        
        # Update result display
        self.result_label.config(
            text=f"Prediction: {result}",
            fg=color
        )
        self.confidence_label.config(
            text=f"Confidence: {max(normal_prob, pneumonia_prob):.2%}"
        )
    
    def predict_image(self):
        """Make prediction on the selected image."""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first!")
            return
        
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(self.current_image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            self.current_probabilities = probabilities[0].cpu().numpy()
        
        # Update prediction display
        self.update_prediction()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    app = PneumoniaPredictor()
    app.run()

if __name__ == '__main__':
    main() 