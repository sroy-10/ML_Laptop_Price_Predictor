import pickle

import numpy as np
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

df = pickle.load(open("df.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Laptop Price Predictor")

st.title("Laptop Price Predictor")

# brand
company = st.selectbox("Brand", sorted(df["Company"].unique()))

# type of laptop
type = st.selectbox("Type", sorted(df["TypeName"].unique()))

# Ram
ram = st.selectbox("RAM(in GB)", sorted(df["Ram"].unique()))

# cpu
cpu = st.selectbox("CPU", sorted(df["CpuBrand"].unique()))

# memory
hdd = st.selectbox("HDD(in GB)", [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD(in GB)", [0, 8, 128, 256, 512, 1024])

# Gpu
gpu = st.selectbox("GPU", df["Gpu"].unique())

# OS
os = st.selectbox("OS", sorted(df["os"].unique()))

# screen size
screen_size = st.selectbox(
    "Screen Size (in inch)",
    [10.1, 11.6, 12.3, 12.5, 13.3, 13.5, 14, 15.6, 17.3, 18.4],
)

# resolution
resolution = st.selectbox(
    "Screen Resolution",
    [
        "1920x1080",
        "1366x768",
        "1600x900",
        "3840x2160",
        "3200x1800",
        "2880x1800",
        "2560x1600",
        "2560x1440",
        "2304x1440",
    ],
)

# processor speed
processor_speed = st.number_input(
    label="Processor speed (in GHz)", min_value=0.9, step=0.01
)

# Screen Type
col_set1 = st.columns(4)
with col_set1[0]:
    # Touchscreen
    touchscreen = st.radio("Touchscreen", ("No", "Yes"))
with col_set1[1]:
    ips = st.radio("IPS Panel", ("No", "Yes"))
with col_set1[2]:
    _4k = st.radio("4K", ("No", "Yes"))
with col_set1[3]:
    fullHd = st.radio("Full HD", ("No", "Yes"))

col_set2 = st.columns(3)
with col_set2[0]:
    ultraHd = st.radio("Ultra HD", ("No", "Yes"))
with col_set2[1]:
    retinaDis = st.radio("Retina Display", ("No", "Yes"))
with col_set2[2]:
    quadHd = st.radio("Quad HD+", ("No", "Yes"))

# weight
weight = st.number_input("Weight of the Laptop (in Kg)", value=1.0)

if st.button("Predict Price"):
    # get the radio button value
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0
    _4k = 1 if _4k == "Yes" else 0
    fullHd = 1 if fullHd == "Yes" else 0
    ultraHd = 1 if ultraHd == "Yes" else 0
    retinaDis = 1 if retinaDis == "Yes" else 0
    quadHd = 1 if quadHd == "Yes" else 0

    # query
    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size
    query = np.array(
        [
            company,
            type,
            ram,
            gpu,
            weight,
            touchscreen,
            ips,
            _4k,
            fullHd,
            ultraHd,
            retinaDis,
            quadHd,
            ppi,
            cpu,
            float(processor_speed),
            ssd,
            hdd,
            os,
        ]
    )
    query = query.reshape(1, 18).astype(object)
    predicted_price = int(np.exp(model.predict(query)[0]))
    st.subheader("The predicted price of this configuration")
    st.title("{:,.0f}".format(predicted_price))
