# ============================================================
# App de Streamlit: ¿La conexión de red es normal o sospechosa?
# Modelos: 4 kernels de SVM (lineal, polinomial, RBF, sigmoide)
#          + árbol de decisión.
# La persona escribe el tráfico (Mbps) y la latencia (ms) y los
# modelos clasifican la conexión como Normal (0) o Sospechosa (1).
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------------
st.set_page_config(page_title="Conexiones sospechosas", layout="centered")
st.title("Clasificación de conexiones de red - Vianey Diaz - ISC - Campus Comayagua")

# Colores y nombres de las clases (iguales a los del cuaderno)
COLORES = ["#0891b2", "#f97316"]
NOMBRES_CLASE = {0: "Normal", 1: "Sospechosa"}


# ------------------------------------------------------------
# Función auxiliar: busca un archivo en la raíz o en las carpetas
# models/ y outputs/ (así funciona sin importar dónde lo subas)
# ------------------------------------------------------------
def buscar_archivo(nombre):
    rutas = [nombre, f"models/{nombre}", f"outputs/{nombre}"]
    for ruta in rutas:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(f"No se encontró el archivo: {nombre}")


# ------------------------------------------------------------
# Carga de los modelos y la metadata (una sola vez gracias al caché)
# ------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    modelos = joblib.load(buscar_archivo("modelos_kernels_conexiones.pkl"))
    with open(buscar_archivo("model_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return modelos, metadata


# ------------------------------------------------------------
# Carga de la base de datos (para la gráfica y la vista previa)
# ------------------------------------------------------------
@st.cache_data
def cargar_base():
    return pd.read_csv(buscar_archivo("conexiones_red.csv"))


modelos, metadata = cargar_modelos()
df = cargar_base()

# Datos guardados en la metadata desde Colab
variables = metadata["variables_usadas"]            # ['trafico_mbps', 'latencia_ms']
seleccionado = metadata["modelo_seleccionado"]      # el mejor según validación cruzada
rangos = metadata["rangos"]                         # mínimo y máximo de cada variable

st.caption(metadata["nombre_modelo"])

# Aviso si la versión de scikit-learn no coincide con la de Colab
if sklearn.__version__ != metadata.get("version_sklearn"):
    st.warning(
        f"Los modelos se entrenaron con scikit-learn {metadata.get('version_sklearn')} "
        f"pero la app usa {sklearn.__version__}. Ajuste requirements.txt."
    )

# ------------------------------------------------------------
# Formulario: la persona escribe los datos de la conexión
# ------------------------------------------------------------
st.subheader("Datos de la conexión")

with st.form("datos"):
    col1, col2 = st.columns(2)

    # Los límites salen de la metadata; se deja margen para probar valores extremos
    trafico = col1.number_input(
        "Tráfico (Mbps)",
        min_value=0.0,
        max_value=float(rangos["trafico_mbps"][1]) * 2,
        value=100.0,
        step=1.0,
    )
    latencia = col2.number_input(
        "Latencia (ms)",
        min_value=0.0,
        max_value=float(rangos["latencia_ms"][1]) * 2,
        value=100.0,
        step=1.0,
    )

    # Modelo que se usa para el resultado principal y la frontera
    modelo_elegido = st.selectbox(
        "Modelo para el resultado principal",
        metadata["orden_modelos"],
        index=metadata["orden_modelos"].index(seleccionado),
    )

    enviar = st.form_submit_button("Evaluar")

# ------------------------------------------------------------
# Predicción (solo cuando se presiona "Evaluar")
# ------------------------------------------------------------
if enviar:
    # Misma estructura que en el cuaderno: DataFrame con los nombres de columnas
    nueva = pd.DataFrame({"trafico_mbps": [trafico], "latencia_ms": [latencia]})

    # Resultado principal con el modelo elegido
    pred = int(modelos[modelo_elegido].predict(nueva)[0])

    st.subheader("Resultado")
    if pred == 1:
        st.error(f"Conexión SOSPECHOSA ({modelo_elegido})")
    else:
        st.success(f"Conexión NORMAL ({modelo_elegido})")

    if modelo_elegido == seleccionado:
        st.caption("Este es el modelo seleccionado por validación cruzada.")

    # Predicción de los 5 modelos (igual que el ciclo for del cuaderno)
    st.write("Predicción de cada modelo")
    tabla = pd.DataFrame(
        [
            {
                "Modelo": nombre,
                "Predicción": NOMBRES_CLASE[int(m.predict(nueva)[0])],
            }
            for nombre, m in modelos.items()
        ]
    ).set_index("Modelo")
    st.dataframe(tabla, width="stretch")

    # --------------------------------------------------------
    # Gráfica: frontera de decisión del modelo elegido + el punto nuevo
    # --------------------------------------------------------
    st.write(f"Frontera de decisión: {modelo_elegido}")

    x_min = min(df["trafico_mbps"].min(), trafico) - 4
    x_max = max(df["trafico_mbps"].max(), trafico) + 4
    y_min = min(df["latencia_ms"].min(), latencia) - 4
    y_max = max(df["latencia_ms"].max(), latencia) + 4

    # Rejilla de puntos para pintar las zonas de cada clase
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    rejilla = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=variables)
    zonas = modelos[modelo_elegido].predict(rejilla).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.contourf(xx, yy, zonas, levels=[-0.5, 0.5, 1.5], colors=COLORES, alpha=0.15)

    # Puntos de la base de datos, por clase
    for clase, nombre in NOMBRES_CLASE.items():
        d = df[df["clase"] == clase]
        ax.scatter(
            d["trafico_mbps"], d["latencia_ms"],
            c=COLORES[clase], label=nombre, alpha=0.75,
            edgecolors="white", s=30,
        )

    # La conexión que se está probando
    ax.scatter(
        [trafico], [latencia], marker="*", s=350, c="black",
        edgecolors="white", linewidths=1.2, label="Conexión nueva", zorder=5,
    )
    ax.set(xlabel="Tráfico (Mbps)", ylabel="Latencia (ms)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------
# Información de los modelos (viene de la metadata)
# ------------------------------------------------------------
st.divider()
st.subheader("Comparación de modelos")
st.write(f"Modelo seleccionado por validación cruzada: **{seleccionado}**")

st.write("Validación cruzada (5 particiones, solo datos de entrenamiento)")
st.dataframe(
    pd.DataFrame(metadata["tabla_validacion_cruzada"]).set_index("Modelo"),
    width="stretch",
)

st.write("Desempeño en los datos de prueba")
tabla_prueba = pd.DataFrame(metadata["tabla_prueba"]).set_index("Modelo")
st.dataframe(tabla_prueba, width="stretch")
st.bar_chart(tabla_prueba[["Accuracy", "F1"]])

# Vista rápida de la base de datos
st.write("Vista rápida de la base de datos")
st.dataframe(df.head(20), width="stretch")
