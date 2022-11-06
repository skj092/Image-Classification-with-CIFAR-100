# Building an Streamlit App for the CIFAR-10 Dataset

# Importing the necessary libraries
import streamlit as st
import torch, torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from model import CNN 
import config
from config import load_checkpoint
import cv2 

# Title of the App
st.title("CIFAR-10 Image Classifier")
st.write("This is a simple image classification web app to predict the class of an image from the CIFAR-10 dataset.")

# Loading the model
model = CNN() 
load_checkpoint(torch.load(config.CHECKPOINT_FILE), model)
print('model loaded successfully')
model.to(config.device)

# Loading the classes
classes = config.CLASSES

# Loading the image jpg or png
image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
    img = Image.open(image_file).convert('RGB')
    st.text("Original Image")
    st.image(img)

    # Transforming the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img = img.unsqueeze(0)

    # Predicting the class of the image
    with torch.no_grad():
        yb = model(img)
        _, preds  = torch.max(yb, dim=1)
        st.write(f'Predicted class: {classes[preds[0]]}')
        st.write(f'Confidence: {torch.softmax(yb, dim=1)[0][preds[0]]*100:.2f}%')