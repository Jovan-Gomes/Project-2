import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image

# Load models
pneumonia_model = load_model("D:\ChestXRay_Dataset\pneumonia\Pneumonia.h5")
cardiomegaly_model = load_model("D:\ChestXRay_Dataset\cardiomegaly.h5")
lung_cancer_model = load_model("D:\ChestXRay_Dataset\lung_cancer\lung_cancer_model.h5")

# Image Preprocessing Functions
def preprocess_for_pneumonia(img):
    img = img.convert("RGB")  # Ensure 3 color channels
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # VGG16 expects input in a specific format
    return img

def preprocess_for_cardiomegaly(img):
    img = np.array(img)  # Convert PIL Image to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    img = cv2.resize(img, (64, 64))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img.astype(np.float32)

def preprocess_for_lung_cancer(img):
    """Preprocess image for Lung Cancer model (350x350)."""
    img = img.convert("RGB")  # Ensure 3 color channels (removes alpha channel if present)
    img = img.resize((350, 350))  
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize
    return img


# Streamlit UI
st.title("🩺 Radiology Analysis Tool")

# Step 1: Choose type of scan
st.subheader("Select the type of scan:")
scan_type = st.radio("", ["X-ray", "CT scan"])

if scan_type == "X-ray":
    st.subheader("📂 Upload Chest X-ray Image")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded X-ray", use_container_width=True)

        img_pneumonia = preprocess_for_pneumonia(image_pil)
        img_cardiomegaly = preprocess_for_cardiomegaly(image_pil)

        pneumonia_pred = pneumonia_model.predict(img_pneumonia)[0][1]  # Using [0][1]
        cardiomegaly_pred = cardiomegaly_model.predict(img_cardiomegaly)[0][1]  # Using [0][1]

        st.subheader("📝 Diagnosis:")
        if pneumonia_pred > 0.5:
            st.warning(f"⚠️ Pneumonia detected (Confidence: {pneumonia_pred:.2f})")
        if cardiomegaly_pred > 0.5:
            st.warning(f"⚠️ Cardiomegaly detected (Confidence: {cardiomegaly_pred:.2f})")
        if pneumonia_pred <= 0.5 and cardiomegaly_pred <= 0.5:
            st.success("✅ No significant signs of Pneumonia or Cardiomegaly detected.")

        st.subheader("📖 About the Conditions")
        st.write("Pneumonia is a lung infection that causes inflammation and fluid build-up in the lungs.")
        st.write("   - Symptoms include shortness of breath, fever and chest pain.")
        st.write("   - Look for bluish lips or fingernails.")
        st.write("Cardiomegaly refers to an enlarged heart, which can be caused by heart disease.")
        st.write("   - Symptoms include irregular hearthbeats, chest pain and dizziness.")
        st.write("   - Look for swelling in the legs, ankles or feet.")

elif scan_type == "CT scan":
    st.subheader("📂 Upload Lung CT Scan Image")
    uploaded_file = st.file_uploader("Choose a CT scan image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded CT Scan", use_container_width=True)
        
        img_lung_cancer = preprocess_for_lung_cancer(image_pil)
        lung_cancer_pred = lung_cancer_model.predict(img_lung_cancer)
        predicted_class = np.argmax(lung_cancer_pred[0])
        class_labels = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
        predicted_label = class_labels[predicted_class]

        st.subheader("📝 Diagnosis:")
        if predicted_label != 'Normal':
            st.error(f"⚠️ Lung Cancer Detected (Type: {predicted_label} , Confidence: {lung_cancer_pred[0][predicted_class]:.2f})")
        else:
            st.success("✅ No Lung Cancer Detected.")
        
        st.subheader("📖 About Lung Cancer Types")
        st.write("1. **Adenocarcinoma**: A type of lung cancer that begins in the glandular cells.")
        st.write("2. **Large Cell Carcinoma**: A type of lung cancer that is fast-growing and aggressive.")
        st.write("3. **Squamous Cell Carcinoma**: A type of lung cancer that begins in the squamous cells.")
    