import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, 'rb') as img_file:
        base64_image = base64.b64encode(img_file.read()).decode()
    return base64_image

# Path to your image
image_path = "C:/Users/sandh/OneDrive/Desktop/dataset-card.jpg"  # Adjust the path accordingly
base64_image = get_base64_image(image_path)

# CSS for background image and text color
background_style = f"""
    <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{base64_image}');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
            height: 100vh;  /* Ensures the background image covers the entire viewport height */
            width: 100vw;   /* Ensures the background image covers the entire viewport width */
            color: black;   /* Text color */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        .stTextInput, .stTextArea, .stSelectbox, .stNumberInput, .stButton {{
            color: black !important; /* Input and button text color */
        }}
        .mandatory::after {{
            content: " *";
            color: red;
        }}
    </style>
"""

# Injecting the background image CSS into the Streamlit app
st.markdown(background_style, unsafe_allow_html=True)

# Main content of the app
st.title("Laptop Price Predictor")

# Load the model and data (replace with your actual loading logic)
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Dropdowns and number input for user inputs
company = st.selectbox('Brand', df['Company'].unique(), key='company')
type_name = st.selectbox('Type', df['TypeName'].unique(), key='type_name')
ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], key='ram')
weight = st.number_input('Weight of the laptop', min_value=0.0, value=1.0, key='weight')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'], key='touchscreen')
ips = st.selectbox('IPS', ['No', 'Yes'], key='ips')
screen_size = st.number_input('Screen Size', min_value=0.0, value=13.3, key='screen_size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2560x1600', '2560x1440', '2304x1440'], key='resolution')
cpu = st.selectbox('CPU', df['Cpu brand'].unique(), key='cpu')
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048], key='hdd')
sdd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024], key='sdd')
gpu = st.selectbox('GPU', df['Gpu brand'].unique(), key='gpu')
os = st.selectbox('OS', df['os'].unique(), key='os')

# Button to predict price
if st.button('Predict Price'):
    try:
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        if screen_size == 0:
            raise ValueError("Screen size cannot be zero.")
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        query = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen_val],
            'IPS': [ips_val],
            'ppi': [ppi],
            'Cpu brand': [cpu],
            'HDD': [hdd],
            'SDD': [sdd],
            'Gpu brand': [gpu],
            'os': [os]
        })

        prediction = np.exp(pipe.predict(query)[0])
        st.markdown(f"<h2 style='color: black;'>The Predicted Price of this Configuration is ₹{int(prediction):,}</h2>", unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error predicting price: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
