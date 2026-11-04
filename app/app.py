''' Interfaz web para segmentar fotos con los modelos entrenados '''

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

# Los scripts del proyecto están en src/, se agrega esa carpeta para poder importarlos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import MODELS_DIR
from explore_data import load_class_names
from predict import COLORS, colorize, find_real_mask, image_metrics, load_image, predict_mask


@st.cache_resource
def get_model(experiment):
    ''' Carga el modelo una sola vez y lo reutiliza en cada predicción '''

    return tf.keras.models.load_model(MODELS_DIR / f"{experiment}.keras", compile=False)


@st.cache_resource
def get_names():
    ''' Carga los nombres de las 150 clases una sola vez '''

    return load_class_names()


def legend_html(classes, counts, names, total):
    ''' Arma la leyenda con el color, el nombre y el porcentaje de la foto que ocupa cada clase '''

    rows = []
    for class_id, count in zip(classes, counts):
        r, g, b = COLORS[class_id]
        name = names.get(class_id + 1, str(class_id + 1))
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
            f"<span style='width:14px;height:14px;background:rgb({r},{g},{b});display:inline-block;border-radius:3px'></span>"
            f"<span>{name}: {count / total * 100:.1f}%</span></div>"
        )

    return "".join(rows)


def prediction_tab(experiment, names):
    ''' Sección para subir una foto y ver su segmentación '''

    uploaded = st.file_uploader("Sube una foto", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Sube una foto para ver la máscara que predice el modelo.")
        return

    original, image = load_image(uploaded)
    with st.spinner("Segmentando la foto..."):
        predicted = predict_mask(get_model(experiment), image, original.shape)

    # Si la foto es del dataset se busca su máscara real por el nombre del archivo
    real = find_real_mask(uploaded.name)

    overlay = (0.5 * original + 0.5 * colorize(predicted)).astype(np.uint8)
    columns = st.columns(4 if real is not None else 3)
    columns[0].image(original, caption="Foto original", width="stretch")
    columns[1].image(colorize(predicted), caption="Máscara predicha", width="stretch")
    columns[2].image(overlay, caption="Predicción sobre la foto", width="stretch")

    if real is not None:
        columns[3].image(colorize(real), caption="Máscara real", width="stretch")
        accuracy, miou = image_metrics(real, predicted)
        metric_columns = st.columns(2)
        metric_columns[0].metric("Exactitud por píxel", f"{accuracy:.3f}")
        metric_columns[1].metric("mIoU de la imagen", f"{miou:.3f}")
    else:
        st.caption("Esta foto no pertenece al dataset, por eso no hay máscara real ni métricas.")

    # Clases detectadas ordenadas de la que ocupa más área a la que ocupa menos
    classes, counts = np.unique(predicted, return_counts=True)
    order = np.argsort(counts)[::-1][:15]
    st.subheader("Clases detectadas")
    st.markdown(legend_html(classes[order], counts[order], names, predicted.size), unsafe_allow_html=True)


def main():
    ''' Arma la página con la barra lateral y la sección de predicción '''

    st.set_page_config(page_title="Segmentación de escenas", layout="wide")
    st.title("Segmentación semántica de escenas con U-Net")

    # Los modelos disponibles son los archivos .keras que hay en models/
    experiments = sorted(path.stem for path in MODELS_DIR.glob("*.keras"))
    if not experiments:
        st.error("No hay modelos en models/. Primero hay que entrenar uno con src/train.py.")
        return

    experiment = st.sidebar.selectbox("Modelo", experiments, index=len(experiments) - 1)
    st.sidebar.caption("U-Net entrenada desde cero con ADE20K, 150 clases.")

    prediction_tab(experiment, get_names())


main()
