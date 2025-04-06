# Pneumonia Detection System

A deep learning-based system for detecting pneumonia in chest X-ray images.

## Features

- Upload and analyze chest X-ray images
- Real-time prediction with confidence scores
- Adjustable prediction threshold
- Modern, user-friendly interface
- Detailed recommendations based on results

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model on your dataset:

```
python src/train.py
```

This will:
- Load and preprocess the training data
- Train the CNN model
- Save the trained model to `models/pneumonia_model.pth`
- Generate a training history plot

### Running the Streamlit App

To start the Streamlit app:

```
streamlit run src/app.py
```

This will open a web browser with the application interface where you can:
- Upload chest X-ray images
- Get instant predictions
- Adjust the prediction threshold
- View detailed analysis results

## Dataset Structure

The expected dataset structure is:

```
data/
├── train/
│   ├── NORMAL/
│   │   └── *.jpeg
│   └── PNEUMONIA/
│       └── *.jpeg
└── val/
    ├── NORMAL/
    │   └── *.jpeg
    └── PNEUMONIA/
        └── *.jpeg
```

## Model Architecture

The system uses a custom CNN architecture with:
- 4 convolutional blocks with batch normalization
- Global average pooling
- Fully connected layers with dropout
- Optimized for chest X-ray image analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Inspired by research in medical image analysis 