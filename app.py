import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.image('Logo.png')
modulo = st.sidebar.selectbox("Seleccione un Módulo", ["Home","Carga de Dataset"])
st.sidebar.image('logo_explorar.png')
if modulo == "Home":
        st.title("🏦 Análisis Exploratorio de Datos: BankMarketing")
else:
    st.title("Carga del dataset")
