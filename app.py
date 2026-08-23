import io
import streamlit as st
import pandas as pd
import numpy as np

st.sidebar.image('Logo.png')
modulo = st.sidebar.selectbox("Seleccione un Módulo", ["Home","Carga de Dataset"])
st.sidebar.image('Python_logo.png')
if modulo == "Home":
    st.title("🏦 Análisis Exploratorio de Datos: BankMarketing")
else:
    st.title("Carga del dataset")
