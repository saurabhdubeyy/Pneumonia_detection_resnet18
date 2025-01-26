import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Load the saved model
@st.cache_resource
def load_model():
    # Define the model architecture (must match the saved model)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Update final layer for binary classification
    model.load_state_dict(torch.load('pneumonia-classifier.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit UI
st.title("Pneumonia Classifier")
st.write("Upload a chest X-ray image to check if it shows signs of pneumonia.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Display the result
    class_names = ['Normal', 'Pneumonia']
    st.write(f"Prediction: **{class_names[predicted.item()]}**")
