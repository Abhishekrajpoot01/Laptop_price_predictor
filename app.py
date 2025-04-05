import streamlit as st
import pickle
import numpy as np

# Load model and data
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# Page configuration
st.set_page_config(page_title="Laptop Price Predictor 💻", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #2c3e50;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #0e4f8f;
        text-align: center;
    }
    .stButton > button {
        background-color: #0e4f8f;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 10px;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main'><h1>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)

# Form layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('🏢 Brand', df['Company'].unique())
    type = st.selectbox('🧰 Type', df['TypeName'].unique())
    ram = st.selectbox('🧠 RAM (in GB)', [2,4,6,8,12,16,24,32,64])
    weight = st.number_input('⚖️ Weight of the Laptop')
    touchscreen = st.selectbox('🖱️ Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('🎨 IPS Display', ['No', 'Yes'])

with col2:
    screen_size = st.slider('📐 Screen Size (inches)', 10.0, 18.0, 13.0)
    resolution = st.selectbox('🖥️ Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
    cpu = st.selectbox('⚙️ CPU Brand', df['Cpu brand'].unique())
    hdd = st.selectbox('💾 HDD (in GB)', [0,128,256,512,1024,2048])
    ssd = st.selectbox('🔋 SSD (in GB)', [0,8,128,256,512,1024])
    gpu = st.selectbox('🎮 GPU Brand', df['Gpu brand'].unique())
    os = st.selectbox('🖥️ Operating System', df['os'].unique())

# Prediction logic
if st.button('🚀 Predict Price'):
    try:
        touchscreen_val = 1 if touchscreen == 'Yes' else 0
        ips_val = 1 if ips == 'Yes' else 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        query = np.array([company, type, ram, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        predicted_price = int(np.exp(pipe.predict(query)[0]))

        st.success(f"💰 The estimated price of this laptop is **₹ {predicted_price:,}**")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

# Close the div
st.markdown("</div>", unsafe_allow_html=True)
