# Pneumonia Detection from Chest X-Ray Images

A deep learning application that detects pneumonia from chest X-ray images using PyTorch and ResNet18.

![Pneumonia Detection](https://i.imgur.com/jZqpV51.png)

## Overview

This application uses a convolutional neural network (ResNet18) trained on chest X-ray images to classify whether a patient has pneumonia or not. The model has been trained on a dataset of labeled chest X-ray images, and the application provides an easy-to-use web interface built with Streamlit.

## Features

- Upload and analyze chest X-ray images
- Real-time classification (Normal vs. Pneumonia)
- Simple and intuitive user interface
- Pre-trained model for immediate use

## Technology Stack

- **Python 3.9+**
- **PyTorch**: Deep learning framework for model training and inference
- **Streamlit**: For the web application interface
- **ResNet18**: Pre-trained CNN architecture, fine-tuned for pneumonia detection
- **PIL (Pillow)**: For image processing

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python -m streamlit run app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://localhost:8501
   ```

## Usage

1. Launch the application using the instructions above
2. Click on "Choose an image file" to upload a chest X-ray image (JPG, JPEG, or PNG)
3. The application will display the uploaded image and provide a classification result
4. The result will indicate whether the X-ray shows signs of pneumonia or appears normal

## Model Details

The application uses a ResNet18 model pre-trained on ImageNet and fine-tuned on a dataset of labeled chest X-ray images. The model has been trained to identify visual patterns associated with pneumonia in chest X-rays.

- **Architecture**: ResNet18
- **Classes**: Normal, Pneumonia
- **Input size**: 224x224 pixels
- **Pre-processing**: Resizing, normalization

## Project Structure

```
pneumonia-detection/
├── app.py                  # Streamlit application
├── pneumonia-classifier.pth # Trained model weights
├── requirements.txt        # Python dependencies
├── Notebook.ipynb          # Model training notebook
└── README.md               # This file
```

## Deployment Options

### Streamlit Cloud (Recommended)
1. Create a GitHub repository with your code
2. Sign up at https://streamlit.io/cloud
3. Connect to your GitHub repository and deploy

### Hugging Face Spaces
1. Create an account at https://huggingface.co/
2. Create a new Space with Streamlit template
3. Push your code to the Hugging Face repository

### Other Platforms
The application can also be deployed on services like Render, Heroku, or any platform that supports Python applications.

## Future Improvements

- Add confidence scores for predictions
- Support for batch processing of multiple images
- Visualization of model activation maps
- Additional model architectures for comparison
- Mobile-friendly interface

## License

[MIT License](LICENSE)

## Acknowledgements

- The model was trained using the Chest X-Ray Images (Pneumonia) dataset from Kaggle
- Thanks to the PyTorch and Streamlit communities for their excellent documentation and tools 