import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os
import torch.nn as nn
import torchvision.models as models

# Установка устройства
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Классы
CLASS_NAMES = ['benign', 'malignant']

# Функция для загрузки модели
@st.cache_resource
def load_model():
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Предиктор
def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    return CLASS_NAMES[pred_idx], confidence

# Интерфейс
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.title("🩺 Skin Cancer Classifier")
st.markdown("### Определяет доброкачественные и злокачественные опухоли на основе изображения.")

# Информация о модели
with st.expander("ℹ️ О модели"):
    st.markdown("""
    - **Модель**: EfficientNet-B0
    - **Классы**: доброкачественный / злокачественный
    - **Метрики**:
        - Accuracy: ~0.90
        - Precision: ~0.85
        - Recall: ~0.94
        - F1 Score: ~0.89
        - ROC AUC: ~0.98
    - **Что значат эти метрики?**
        - **Accuracy** — общая точность распознавания.
        - **Precision** — доля правильных положительных предсказаний.
        - **Recall** — способность находить все реальные положительные случаи.
        - **F1** — гармоническое среднее между precision и recall.
        - **ROC AUC** — качество бинарной классификации.
    """)

# Выбор источника изображения
option = st.selectbox("Выберите источник изображения:", ("Загрузить изображение", "URL изображения", "Сделать фото"))

image = None

if option == "Загрузить изображение":
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "URL изображения":
    url = st.text_input("Введите URL изображения:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("Не удалось загрузить изображение по указанному URL.")

elif option == "Сделать фото":
    camera_image = st.camera_input("Сделайте фото")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# Предсказание
if image:
    st.image(image, caption="Ваше изображение", use_column_width=True)
    label, confidence = predict_image(image)
    st.success(f"Предсказание: **{label}**")
    st.info(f"Уверенность: **{confidence:.2%}**")