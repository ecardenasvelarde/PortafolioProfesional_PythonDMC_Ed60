import io
import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------
st.set_page_config(
    page_title="EDA - BankMarketing",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------------------------------------------
# VARIABLES GLOBALES (datos del autor)
# -----------------------------------------------------------------
AUTOR_NOMBRE = "Tu Nombre Completo"
AUTOR_CURSO = "Especialización en Python for Analytics - DMC Ed.60"
AUTOR_ANIO = "2026"

# -----------------------------------------------------------------
# ESTADO DE LA SESIÓN
# -----------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None


# -----------------------------------------------------------------
# FUNCIÓN PERSONALIZADA (requisito Ítem 2)
# -----------------------------------------------------------------
def identificar_tipo_variable(serie: pd.Series) -> str:
    """Clasifica una columna como 'Numérica' o 'Categórica'."""
    if pd.api.types.is_numeric_dtype(serie):
        return "Numérica"
    else:
        return "Categórica"


# -----------------------------------------------------------------
# CLASE DataAnalyzer (POO)
# -----------------------------------------------------------------
class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def info_general(self) -> str:
        """Ítem 1: equivalente a df.info(), capturado como texto."""
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def conteo_nulos(self) -> pd.Series:
        """Ítem 1: cantidad de valores nulos por columna."""
        return self.df.isnull().sum()

    def clasificar_variables(self) -> pd.DataFrame:
        """Ítem 2: clasifica cada columna usando la función personalizada."""
        registros = []
        for columna in self.df.columns:
            registros.append({
                "Variable": columna,
                "Tipo (dtype)": str(self.df[columna].dtype),
                "Clasificación": identificar_tipo_variable(self.df[columna])
            })
        return pd.DataFrame(registros)


# -----------------------------------------------------------------
# SIDEBAR: logos + navegación (SOLO 2 MÓDULOS)
# -----------------------------------------------------------------
st.sidebar.image('DMC.png')
modulo = st.sidebar.selectbox("Seleccione un Módulo", ["Home", "Carga de Dataset"])
st.sidebar.image('Python_logo.png')

# -----------------------------------------------------------------
# MÓDULO 1: HOME
# -----------------------------------------------------------------
if modulo == "Home":
    st.title("🏦 Análisis Exploratorio de Datos: BankMarketing")
    st.markdown(
        f"""
        ### 🎯 Objetivo del análisis
        Esta aplicación explora el dataset de una campaña de marketing
        bancario para entender **qué factores se relacionan con que un
        cliente acepte o no un depósito a plazo**.

        ### 👤 Datos del autor
        - **Nombre:** {AUTOR_NOMBRE}
        - **Curso:** {AUTOR_CURSO}
        - **Año:** {AUTOR_ANIO}

        ### 📊 Sobre el dataset
        `BankMarketing.csv` contiene **41,188 registros** y **21 variables**.
        La variable objetivo es `y`: si el cliente aceptó ("yes") o no
        ("no") la campaña.

        ### 🛠️ Tecnologías utilizadas
        Python · Pandas · NumPy · Streamlit
        """
    )

# -----------------------------------------------------------------
# MÓDULO 2: CARGA DE DATASET  ->  incluye el EDA completo debajo
# -----------------------------------------------------------------
else:
    st.title("📂 Carga del Dataset")
    st.write("Sube el archivo `BankMarketing.csv` para habilitar el análisis.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo, sep=";")
            st.session_state.df = df
            st.success(f"✅ Archivo cargado correctamente: **{archivo.name}**")
        except Exception as e:
            st.error(f"❌ Ocurrió un error al leer el archivo: {e}")
            st.session_state.df = None

    # ---------------------------------------------------------
    # Si NO hay datos cargados, avisamos y no mostramos nada más.
    # ---------------------------------------------------------
    if st.session_state.df is None:
        st.info("⬆️ Aún no se ha cargado ningún archivo.")
        st.stop()

    df = st.session_state.df

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columnas", f"{df.shape[1]:,}")

    st.divider()

    # ---------------------------------------------------------
    # A partir de aquí: Análisis Exploratorio de Datos (EDA)
    # Vive DENTRO del módulo "Carga de Dataset", como pediste.
    # ---------------------------------------------------------
    st.header("📊 Análisis Exploratorio de Datos (EDA)")

    analyzer = DataAnalyzer(df)  # instanciamos la clase una vez

    tabs = st.tabs([
        "1. Info general", "2. Tipos de variable", "3. Estadísticas",
        "4. Nulos", "5. Distribuciones", "6. Categóricas",
        "7. Bivariado num-cat", "8. Bivariado cat-cat",
        "9. Interactivo", "10. Hallazgos"
    ])

    # ---------- Ítem 1: Información general del dataset ----------
    with tabs[0]:
        st.subheader("Información general del dataset")
        st.markdown(
            "Usamos `.info()` para ver de un vistazo cuántas filas tiene "
            "cada columna, su tipo de dato, y si hay valores nulos."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Salida de `df.info()`**")
            st.text(analyzer.info_general())
        with c2:
            st.markdown("**Conteo de valores nulos por columna**")
            st.dataframe(analyzer.conteo_nulos())

        st.info(
            "💡 A simple vista no hay valores nulos (`NaN`), pero en el "
            "Ítem 4 vamos a revisar si existen valores como `'unknown'` "
            "que funcionan como nulos disfrazados."
        )

    # ---------- Ítem 2: Clasificación de variables ----------
    with tabs[1]:
        st.subheader("Clasificación de variables")
        st.markdown(
            "Clasificamos cada columna como **Numérica** o **Categórica** "
            "usando una función personalizada (`identificar_tipo_variable`)."
        )
        clasificacion = analyzer.clasificar_variables()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(clasificacion, use_container_width=True)
        with c2:
            st.markdown("**Resumen**")
            st.dataframe(clasificacion["Clasificación"].value_counts())

    # ---------- Ítems 3 al 10: pendientes ----------
    for i in range(2, 10):
        with tabs[i]:
            st.info("🚧 Este ítem lo construiremos en el siguiente paso.")
