''' Segmenta fotos con un modelo entrenado y guarda una figura por foto con la máscara predicha y las clases detectadas '''

import sys
from pathlib import Path

import matplotlib

# Backend sin ventana: las figuras solo se guardan como archivo
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Patch
from PIL import Image, ImageOps

from config import ANNOTATIONS_DIR, IMG_SIZE, MODELS_DIR, NUM_CLASSES, RESULTS_DIR, SEED, SPLITS
from explore_data import load_class_names

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Número máximo de clases que se muestran en la leyenda, de la que ocupa más área a la que ocupa menos
LEGEND_CLASSES = 10

# Un color fijo por clase; con la semilla siempre salen los mismos colores
COLORS = np.random.default_rng(SEED).integers(0, 256, (NUM_CLASSES, 3), dtype=np.uint8)


def load_image(path):
    ''' Lee una foto con 3 canales y regresa la foto original y la versión preparada para el modelo '''

    # exif_transpose endereza las fotos de celular, que muchas veces se guardan giradas con una etiqueta de orientación
    with Image.open(path) as photo:
        original = np.array(ImageOps.exif_transpose(photo).convert("RGB"))

    # El mismo preprocesamiento del entrenamiento: redimensión bilineal, redondeo a enteros y valores entre 0 y 1
    resized = tf.image.resize(original, (IMG_SIZE, IMG_SIZE), method="bilinear")
    resized = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.float32) / 255.0
    return original, resized


def predict_mask(model, image, original_shape):
    ''' Predice la clase de cada píxel y regresa la máscara al tamaño original de la foto '''

    logits = model(image[tf.newaxis, ...], training=False)
    mask = tf.argmax(logits[0], axis=-1)

    # Vecino más cercano para regresar al tamaño original sin mezclar clases
    mask = tf.image.resize(mask[..., tf.newaxis], original_shape[:2], method="nearest")
    return mask[..., 0].numpy().astype(np.int32)


def find_real_mask(path):
    ''' Busca la máscara real si la foto pertenece al dataset; regresa None si no existe '''

    for split in SPLITS:
        mask_path = ANNOTATIONS_DIR / split / f"{Path(path).stem}.png"
        if mask_path.exists():
            mask = tf.io.decode_png(tf.io.read_file(str(mask_path)), channels=1)[..., 0].numpy().astype(np.int32)

            # Igual que en el entrenamiento: las clases 1 a 150 pasan a 0 a 149 y la clase 0 queda marcada con -1 para ignorarse
            return mask - 1

    return None


def image_metrics(real, predicted):
    ''' Exactitud por píxel y mIoU de una sola imagen, sin contar los píxeles de la clase 0 '''

    valid = real >= 0
    accuracy = np.mean(real[valid] == predicted[valid])

    # El mIoU se promedia sobre las clases que aparecen en la máscara real o en la predicha
    ious = []
    for class_id in np.union1d(np.unique(real[valid]), np.unique(predicted[valid])):
        intersection = np.sum((real == class_id) & (predicted == class_id) & valid)
        union = np.sum(((real == class_id) | (predicted == class_id)) & valid)
        ious.append(intersection / union)

    return accuracy, np.mean(ious)


def colorize(mask):
    ''' Convierte una máscara de clases en una imagen de colores; los píxeles ignorados quedan en negro '''

    colored = COLORS[np.clip(mask, 0, NUM_CLASSES - 1)]
    colored[mask < 0] = 0
    return colored


def save_figure(original, predicted, real, names, output_file):
    ''' Guarda la foto original, la máscara predicha, la máscara sobre la foto y la máscara real si existe '''

    panels = [("Foto original", original), ("Máscara predicha", colorize(predicted))]

    # La máscara encima de la foto, mitad foto y mitad color
    overlay = (0.5 * original + 0.5 * colorize(predicted)).astype(np.uint8)
    panels.append(("Predicción sobre la foto", overlay))

    if real is not None:
        panels.append(("Máscara real", colorize(real)))

    # La altura de la figura sigue la proporción de la foto, más un espacio abajo para la leyenda
    legend_height = 0.45
    height = 4 * original.shape[0] / original.shape[1] + 0.5 + legend_height
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), height))
    for ax, (title, picture) in zip(axes, panels):
        ax.imshow(picture)
        ax.set_title(title)
        ax.axis("off")

    # Leyenda con las clases que ocupan más área en la predicción
    classes, counts = np.unique(predicted, return_counts=True)
    order = np.argsort(counts)[::-1][:LEGEND_CLASSES]
    handles = [Patch(color=COLORS[classes[i]] / 255, label=names.get(classes[i] + 1, str(classes[i] + 1))) for i in order]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)

    fig.tight_layout(rect=(0, legend_height / height, 1, 1))
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def list_images(arguments):
    ''' Junta las fotos indicadas; si se pasa una carpeta, se toman todas las fotos que contiene '''

    images = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            images.append(path)

    return images


def main():
    ''' Segmenta cada foto indicada e imprime las clases detectadas '''

    # Uso: python src/predict.py v1 foto.jpg otra_foto.jpg carpeta_con_fotos/
    if len(sys.argv) < 3:
        sys.exit("Uso: python src/predict.py <experimento> <foto o carpeta> [más fotos o carpetas]")

    experiment = sys.argv[1]
    images = list_images(sys.argv[2:])
    output_dir = RESULTS_DIR / "predicciones" / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    # compile=False porque solo se necesita el modelo para predecir
    model = tf.keras.models.load_model(MODELS_DIR / f"{experiment}.keras", compile=False)
    names = load_class_names()

    for path in images:
        original, image = load_image(path)
        predicted = predict_mask(model, image, original.shape)
        real = find_real_mask(path)

        print(f"\n{path.name}")

        # Porcentaje de la foto que ocupa cada clase detectada
        classes, counts = np.unique(predicted, return_counts=True)
        for i in np.argsort(counts)[::-1][:LEGEND_CLASSES]:
            print(f"  {names.get(classes[i] + 1, classes[i] + 1)}: {counts[i] / predicted.size * 100:.1f}%")

        if real is not None:
            accuracy, miou = image_metrics(real, predicted)
            print(f"  Con máscara real: exactitud {accuracy:.3f}, mIoU {miou:.3f}")
        else:
            print("  Sin máscara real, solo se muestra la predicción")

        output_file = output_dir / f"{path.stem}.png"
        save_figure(original, predicted, real, names, output_file)
        print(f"  Figura guardada en {output_file}")


if __name__ == "__main__":
    main()
