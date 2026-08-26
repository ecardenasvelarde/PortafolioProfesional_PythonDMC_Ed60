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
        - **Nombre:** Erick Eduardo Cárdenas Velarde
        - **Curso:** Especialización en Python for Analytics - DMC Ed.60
        - **Año:** 2026

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

    # ---------- Ítem 5: Distribución de variables numéricas ----------
    with tabs[4]:
        st.subheader("Distribución de variables numéricas")
        st.markdown(
            "Los histogramas nos permiten ver **cómo se reparten los "
            "valores** de una variable: si son simétricos, si están "
            "concentrados en un rango o si tienen una \"cola\" larga "
            "hacia algún lado (sesgo)."
        )

        # ---- Vista general: una grilla con todas las numéricas ----
        st.markdown("**Vista general**")
        num_cols = analyzer.columnas_numericas()
        columnas_por_fila = 3

        for i in range(0, len(num_cols), columnas_por_fila):
            fila_cols = st.columns(columnas_por_fila)
            for j, col_name in enumerate(num_cols[i:i + columnas_por_fila]):
                with fila_cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(df[col_name], kde=True, ax=ax, color="#4C72B0")
                    ax.set_title(col_name, fontsize=10)
                    ax.set_xlabel("")
                    st.pyplot(fig)
                    plt.close(fig)  # liberamos memoria; si no, con muchos gráficos la app se pone lenta

        st.divider()

        # ---- Vista detallada e interactiva ----
        st.markdown("**Vista detallada (elige una variable)**")

        col_sel, col_slider = st.columns(2)
        with col_sel:
            variable_elegida = st.selectbox("Variable numérica:", num_cols, key="hist_var")
        with col_slider:
            bins = st.slider("Número de bins (barras):", min_value=10, max_value=100, value=30, key="hist_bins")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[variable_elegida], bins=bins, kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribución de {variable_elegida}")
        st.pyplot(fig)
        plt.close(fig)

        # Interpretación automática basada en el sesgo (skewness)
        sesgo = df[variable_elegida].skew()
        if sesgo > 1:
            lectura = "sesgada a la **derecha** (cola larga hacia valores altos)"
        elif sesgo < -1:
            lectura = "sesgada a la **izquierda** (cola larga hacia valores bajos)"
        else:
            lectura = "razonablemente **simétrica**"

        st.markdown(
            f"""
            **Interpretación visual:** la variable `{variable_elegida}` tiene un
            coeficiente de sesgo (skewness) de **{sesgo:.2f}**, lo que indica una
            distribución {lectura}.
            """
        )

    # ---------- Ítem 6: Análisis de variables categóricas ----------
    with tabs[5]:
        st.subheader("Análisis de variables categóricas")
        st.markdown(
            "Para las columnas categóricas revisamos **cuántas veces** "
            "aparece cada valor (conteo) y **qué porcentaje** representa "
            "sobre el total (proporción)."
        )

        cat_cols = analyzer.columnas_categoricas()

        col_a, col_b = st.columns([2, 1])
        with col_a:
            var_cat = st.selectbox("Variable categórica:", cat_cols, key="cat_var")
        with col_b:
            mostrar_pct = st.checkbox("Mostrar como porcentaje", value=False, key="cat_pct")

        conteo = df[var_cat].value_counts()
        if mostrar_pct:
            datos_grafico = (conteo / conteo.sum() * 100).round(1)
            etiqueta_x = "Porcentaje (%)"
        else:
            datos_grafico = conteo
            etiqueta_x = "Cantidad"

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(datos_grafico, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=datos_grafico.values, y=datos_grafico.index, ax=ax, color="#55A868")
            ax.set_xlabel(etiqueta_x)
            ax.set_title(f"Distribución de {var_cat}")
            st.pyplot(fig)
            plt.close(fig)

        categoria_top = conteo.idxmax()
        st.markdown(
            f"**Interpretación:** la categoría más frecuente en `{var_cat}` es "
            f"**'{categoria_top}'**, con **{conteo.max():,}** registros "
            f"(**{conteo.max()/conteo.sum()*100:.1f}%** del total)."
        )

    # ---------- Ítem 7: Bivariado (numérico vs categórico) ----------
    with tabs[6]:
        st.subheader("Análisis bivariado: numérico vs categórico")
        st.markdown(
            "Comparamos cómo varía una variable **numérica** según el "
            "resultado de la campaña (`y`), usando boxplots. Esto ayuda a "
            "ver si esa variable realmente distingue a quienes aceptaron "
            "de quienes no."
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x="y", y="age", ax=ax, palette="Set2")
            ax.set_title("age vs y")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x="y", y="duration", ax=ax, palette="Set2")
            ax.set_title("duration vs y")
            st.pyplot(fig)
            plt.close(fig)

        mediana_dur_no = df[df["y"] == "no"]["duration"].median()
        mediana_dur_yes = df[df["y"] == "yes"]["duration"].median()
        mediana_edad_no = df[df["y"] == "no"]["age"].median()
        mediana_edad_yes = df[df["y"] == "yes"]["age"].median()

        st.markdown(
            f"""
            **Interpretación:**
            - `duration`: la mediana es **{mediana_dur_no:.0f}s** para quienes
              NO aceptaron vs **{mediana_dur_yes:.0f}s** para quienes SÍ
              aceptaron. La diferencia es grande: llamadas más largas se
              asocian con mayor aceptación (algo esperable, ya que una
              llamada corta suele terminar en un rechazo rápido).
            - `age`: la mediana es casi igual entre ambos grupos
              ({mediana_edad_no:.0f} vs {mediana_edad_yes:.0f} años), lo que
              sugiere que la edad por sí sola **no es un buen diferenciador**
              de la aceptación.
            """
        )

    # ---------- Ítem 8: Bivariado (categórico vs categórico) ----------
    with tabs[7]:
        st.subheader("Análisis bivariado: categórico vs categórico")
        st.markdown(
            "Cruzamos dos variables categóricas con `pd.crosstab()` para "
            "ver, dentro de cada categoría, qué porcentaje aceptó la "
            "campaña."
        )

        c1, c2 = st.columns(2)
        with c1:
            ct_edu = pd.crosstab(df["education"], df["y"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(5, 4))
            ct_edu.plot(kind="barh", stacked=True, ax=ax, color=["#C44E52", "#55A868"])
            ax.set_title("education vs y (%)")
            ax.set_xlabel("Porcentaje")
            ax.legend(title="y", loc="lower right", fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            ct_contact = pd.crosstab(df["contact"], df["y"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(5, 4))
            ct_contact.plot(kind="barh", stacked=True, ax=ax, color=["#C44E52", "#55A868"])
            ax.set_title("contact vs y (%)")
            ax.set_xlabel("Porcentaje")
            ax.legend(title="y", loc="lower right", fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        pct_cel = ct_contact.loc["cellular", "yes"]
        pct_tel = ct_contact.loc["telephone", "yes"]
        pct_illit = ct_edu.loc["illiterate", "yes"]

        st.markdown(
            f"""
            **Interpretación:**
            - `contact`: contactar por **celular** logra **{pct_cel:.1f}%** de
              aceptación, casi el triple que por **teléfono fijo**
              (**{pct_tel:.1f}%**). Es una señal clara para priorizar el
              canal de contacto.
            - `education`: el grupo "illiterate" tiene la tasa de aceptación
              más alta (**{pct_illit:.1f}%**), aunque es un grupo muy pequeño
              en tamaño — hay que tener cuidado de no sacar conclusiones
              fuertes de un grupo con pocos registros.
            """
        )

    # ---------- Ítem 9: Análisis basado en parámetros seleccionados ----------
    with tabs[8]:
        st.subheader("Análisis interactivo según tus parámetros")
        st.markdown(
            "Elige tú mismo qué variables numéricas comparar (correlación) "
            "y qué variable categórica cruzar contra `y`."
        )

        st.markdown("**a) Matriz de correlación (elige variables numéricas)**")
        num_cols = analyzer.columnas_numericas()
        vars_elegidas = st.multiselect(
            "Variables numéricas a correlacionar:",
            num_cols,
            default=["age", "duration", "campaign"],
            key="multi_corr"
        )

        if len(vars_elegidas) >= 2:
            corr = df[vars_elegidas].corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ Selecciona al menos 2 variables para calcular la correlación.")

        st.divider()

        st.markdown("**b) Aceptación (`y`) según una variable categórica**")
        cat_cols = analyzer.columnas_categoricas()
        cat_cols_sin_y = [c for c in cat_cols if c != "y"]
        var_dinamica = st.selectbox("Variable categórica:", cat_cols_sin_y, key="select_dinamico")

        ct = pd.crosstab(df[var_dinamica], df["y"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        ct.plot(kind="bar", stacked=True, ax=ax, color=["#C44E52", "#55A868"])
        ax.set_title(f"{var_dinamica} vs y (%)")
        ax.set_ylabel("Porcentaje")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)

    # ---------- Ítem 10: Hallazgos clave ----------
    with tabs[9]:
        st.subheader("Hallazgos clave")

        tasa_general = (df["y"] == "yes").mean() * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        df["y"].value_counts().plot(kind="bar", ax=ax, color=["#C44E52", "#55A868"])
        ax.set_title("Distribución general de la variable objetivo (y)")
        ax.set_ylabel("Cantidad de clientes")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            f"""
            **Resumen visual:** de los {len(df):,} clientes contactados en
            la última campaña, solo el **{tasa_general:.1f}%** aceptó el
            depósito a plazo.

            **Insights principales derivados del EDA:**
            1. La **duración de la llamada** es, por lejos, la señal más
               fuerte de aceptación: las llamadas exitosas duran en mediana
               casi 3 veces más que las que terminan en rechazo.
            2. El **canal de contacto** importa: contactar por celular casi
               triplica la tasa de aceptación frente al teléfono fijo.
            3. La **edad** por sí sola no diferencia bien entre clientes que
               aceptan y los que no.
            4. Varias columnas categóricas (`default`, `education`, `housing`,
               `loan`) tienen un porcentaje relevante de valores `'unknown'`
               que conviene tratar con cuidado antes de sacar conclusiones.
            5. Existe un desbalance fuerte en la variable objetivo
               (~89% "no" vs ~11% "yes"), algo importante a tener en cuenta
               si en el futuro se construye un modelo predictivo con estos
               datos.
            """
        )

    # -----------------------------------------------------------------
    # CONCLUSIONES FINALES (sección aparte, fuera de los 10 ítems)
    # -----------------------------------------------------------------
    st.divider()
    st.header("🎯 Conclusiones finales")

    st.markdown(
        """
        1. **El canal de contacto es una palanca accionable de corto plazo.**
           Priorizar campañas por celular sobre teléfono fijo podría mejorar
           directamente la tasa de conversión, sin necesidad de cambiar el
           perfil de cliente contactado.

        2. **La duración de la llamada actúa como indicador de interés, no
           como variable de decisión previa.** Es útil para *entender* qué
           pasó en la campaña anterior, pero no se puede usar para decidir
           a quién llamar, porque solo se conoce después de la llamada.

        3. **La edad del cliente no es un criterio efectivo de segmentación**
           para esta campaña; el equipo comercial no debería priorizar
           contactos en función de rangos etarios.

        4. **La calidad del dato en `default`, `housing` y `loan` es
           limitada** (hasta 20.9% de "unknown" en `default`), por lo que
           cualquier segmentación futura basada en esas variables debe
           interpretarse con cautela.

        5. **La campaña tiene un fuerte desbalance de resultados** (~89% de
           rechazo), lo que sugiere revisar si el guion de venta o el
           público objetivo de la campaña necesitan ajustarse, más que
           solo optimizar el canal de contacto.
        """
    )
