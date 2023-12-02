import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

model = torch.load('Gender.pt', map_location=torch.device('cpu'))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
class_names = {0: 'Male or ผู้ชาย', 1: 'Female or ผู้หญิง'}
def predict(image):
    image_tensor = preprocess(image)
    image = image_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, classes = torch.max(probs, 1)
    return conf.item(), classes.item()

st.title('Welcome to Gender Classification!!')
st.write('This application will help you classify your image as male or female based on the image.')
upload_file = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])
if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button('Predict'):
        conf, classes = predict(image)
        st.write('I think you are :', class_names[classes])
        st.write('With score : ', conf)
