import io
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="EDA - BankMarketing",
    page_icon="🏦"
)

if "df" not in st.session_state:
    st.session_state.df = None

st.sidebar.image('DMC.png')
modulo = st.sidebar.selectbox("Seleccione un Módulo", ["Home", "Carga de Dataset"])
st.sidebar.image('Python_logo.png')
# -----------------------------------------------------------------
# MÓDULO: HOME
# -----------------------------------------------------------------
if modulo == "Home":
    st.title("🏦 Análisis Exploratorio de Datos: BankMarketing")
    st.divider()
    st.markdown(
        f"""
        ### 🎯 Objetivo del análisis
        Esta aplicación explora el dataset de una campaña de marketing
        bancario para entender **qué factores se relacionan con que un
        cliente acepte o no un depósito a plazo**.

        ### 👤 Datos del autor
        - **Nombre:** Erick Eduardo Cárdenas Velarde
        - **Curso:** Especialización en Python for Analytics - DMC Ed.60
        - **Año:** 2026

        ### 📊 Sobre el dataset
        `BankMarketing.csv` contiene **41,188 registros** y **21 variables**,
        con información demográfica de los clientes (edad, trabajo, estado
        civil, educación), datos de contacto de la campaña (canal, mes,
        duración de la llamada) e indicadores económicos del contexto
        (tasa de empleo, índice de precios, etc.). La variable objetivo
        es `y`: si el cliente aceptó ("yes") o no ("no") la campaña.

        ### 🛠️ Tecnologías utilizadas
        Python · Pandas · NumPy · Streamlit
        """
    )

# -----------------------------------------------------------------
# MÓDULO: CARGA DE DATASET
# -----------------------------------------------------------------
else:
    st.title("📂 Carga del Dataset")

    st.write(
        "Sube el archivo `BankMarketing.csv` para habilitar el módulo "
        "de análisis exploratorio."
    )

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            # OJO: este dataset usa punto y coma (;) como separador, no coma.
            df = pd.read_csv(archivo, sep=";")
            st.session_state.df = df
            st.success(f"✅ Archivo cargado correctamente: **{archivo.name}**")
        except Exception as e:
            st.error(f"❌ Ocurrió un error al leer el archivo: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Vista previa del dataset")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filas", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columnas", f"{df.shape[1]:,}")
    else:
        st.info("⬆️ Aún no se ha cargado ningún archivo.")
