import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    # ---------- Ítem 1 ----------
    def info_general(self) -> str:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def conteo_nulos(self) -> pd.Series:
        return self.df.isnull().sum()

    # ---------- Ítem 2 ----------
    def clasificar_variables(self) -> pd.DataFrame:
        registros = []
        for columna in self.df.columns:
            registros.append({
                "Variable": columna,
                "Tipo (dtype)": str(self.df[columna].dtype),
                "Clasificación": identificar_tipo_variable(self.df[columna])
            })
        return pd.DataFrame(registros)

    def columnas_numericas(self) -> list:
        return [c for c in self.df.columns if identificar_tipo_variable(self.df[c]) == "Numérica"]

    def columnas_categoricas(self) -> list:
        return [c for c in self.df.columns if identificar_tipo_variable(self.df[c]) == "Categórica"]

    # ---------- Ítem 3 ----------
    def estadisticas_descriptivas(self) -> pd.DataFrame:
        """
        describe() aplicado solo a las columnas numéricas, transpuesto
        (.T) para que cada FILA sea una variable y cada COLUMNA una
        métrica (mean, std, min, max, etc.) — se lee más fácil así.
        """
        return self.df[self.columnas_numericas()].describe().T

    # ---------- Ítem 4 ----------
    def valores_unknown(self) -> pd.DataFrame:
        """
        Este dataset no tiene NaN reales, pero usa el texto 'unknown'
        como equivalente a un valor faltante en varias columnas
        categóricas. Contamos cuántos 'unknown' hay por columna.
        """
        registros = []
        for columna in self.columnas_categoricas():
            n_unknown = (self.df[columna] == "unknown").sum()
            if n_unknown > 0:
                registros.append({
                    "Variable": columna,
                    "Cantidad 'unknown'": n_unknown,
                    "Porcentaje": round(n_unknown / len(self.df) * 100, 2)
                })
        return pd.DataFrame(registros).sort_values("Cantidad 'unknown'", ascending=False)


# -----------------------------------------------------------------
# SIDEBAR: logos + navegación (2 módulos)
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
        Python · Pandas · NumPy · Streamlit · Matplotlib · Seaborn
        """
    )

# -----------------------------------------------------------------
# MÓDULO 2: CARGA DE DATASET  ->  incluye el EDA completo debajo
# -----------------------------------------------------------------
else:
    st.title("📂 Carga del Dataset")
    st.write("Sube el archivo `BankMarketing.csv` para habilitar el análisis.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])
    df = None

    if archivo is not None:
        try:
            df = pd.read_csv(archivo, sep=";")
            st.success(f"✅ Archivo cargado correctamente: **{archivo.name}**")
        except Exception as e:
            st.error(f"❌ Ocurrió un error al leer el archivo: {e}")
            df = None

    if df is None:
        st.info("⬆️ Aún no se ha cargado ningún archivo.")
        st.stop()

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columnas", f"{df.shape[1]:,}")

    st.divider()

    # ---------------------------------------------------------
    # Análisis Exploratorio de Datos (EDA)
    # ---------------------------------------------------------
    st.header("📊 Análisis Exploratorio de Datos (EDA)")

    analyzer = DataAnalyzer(df)

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
            "Ítem 4 vamos a revisar los valores `'unknown'`, que "
            "funcionan como nulos disfrazados."
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

    # ---------- Ítem 3: Estadísticas descriptivas ----------
    with tabs[2]:
        st.subheader("Estadísticas descriptivas")
        st.markdown(
            "Usamos `.describe()` sobre las columnas **numéricas** para "
            "ver medidas de tendencia central (media, mediana) y de "
            "dispersión (desviación estándar, mínimo, máximo)."
        )

        stats = analyzer.estadisticas_descriptivas()
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

        # Interpretación automática con f-strings, usando datos reales
        edad_media = df["age"].mean()
        edad_mediana = df["age"].median()
        duracion_media = df["duration"].mean()
        duracion_mediana = df["duration"].median()

        st.markdown("**Interpretación básica:**")
        st.markdown(
            f"""
            - La edad promedio de los clientes es **{edad_media:.1f} años**,
              muy cercana a la mediana (**{edad_mediana:.0f} años**), lo que
              sugiere una distribución bastante simétrica.
            - La duración de la llamada tiene una media de
              **{duracion_media:.0f} segundos**, pero una mediana de solo
              **{duracion_mediana:.0f} segundos**. Que la media sea mucho
              mayor que la mediana indica una distribución **sesgada a la
              derecha**: hay llamadas muy largas (outliers) que "jalan" el
              promedio hacia arriba.
            - `pdays` tiene una mediana de 999, un valor que en este
              dataset significa *"el cliente nunca fue contactado antes"* —
              no es un número real de días, hay que tenerlo en cuenta para
              no interpretarlo mal.
            """
        )

    # ---------- Ítem 4: Análisis de valores faltantes ----------
    with tabs[3]:
        st.subheader("Análisis de valores faltantes")

        st.markdown("**Valores nulos (`NaN`) reales:**")
        nulos = analyzer.conteo_nulos()
        if nulos.sum() == 0:
            st.success("✅ El dataset no tiene valores nulos (`NaN`) reales en ninguna columna.")
        else:
            st.dataframe(nulos[nulos > 0])

        st.markdown("---")
        st.markdown(
            "**Valores `'unknown'` (nulos disfrazados):** en encuestas y "
            "campañas comerciales es común que la ausencia de dato se "
            "registre como texto en vez de dejarse vacío. Aquí vemos en "
            "qué columnas categóricas aparece `'unknown'` y con qué "
            "frecuencia."
        )

        unknowns = analyzer.valores_unknown()

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(unknowns, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=unknowns, x="Cantidad 'unknown'", y="Variable", ax=ax, color="#4C72B0")
            ax.set_title("Cantidad de valores 'unknown' por columna")
            st.pyplot(fig)

        peor_columna = unknowns.iloc[0]
        st.markdown(
            f"""
            **Discusión breve:** la columna con más valores `'unknown'` es
            **`{peor_columna['Variable']}`**, con **{int(peor_columna["Cantidad 'unknown'"])}**
            registros (**{peor_columna['Porcentaje']}%** del total). Es un
            porcentaje considerable, así que más adelante habrá que decidir
            si se trata como una categoría más (dejarla como "unknown") o
            si se excluye del análisis según el tipo de pregunta que se
            quiera responder.
            """
        )

    # ---------- Ítems 5 al 10: pendientes ----------
    for i in range(4, 10):
        with tabs[i]:
            st.info("🚧 Este ítem lo construiremos en el siguiente paso.")
