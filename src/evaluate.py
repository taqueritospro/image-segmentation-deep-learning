''' Evalúa un experimento en validación y prueba: métricas generales, IoU de cada clase y su relación con la frecuencia de la clase '''

import sys

import matplotlib

# Backend sin ventana: las gráficas solo se guardan como archivo
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import MODELS_DIR, NUM_CLASSES, RESULTS_DIR
from data import get_datasets
from explore_data import load_class_names

EXPLORATION_FILE = RESULTS_DIR / "exploracion" / "distribucion_clases.csv"


def confusion_and_loss(model, dataset):
    ''' Recorre un conjunto y acumula la matriz de confusión y la loss, sin contar los píxeles con peso 0 '''

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)
    loss_sum, pixel_count = 0.0, 0

    for images, masks, weights in dataset:
        logits = model(images, training=False)
        predicted = tf.argmax(logits, axis=-1).numpy()

        # La loss se calcula igual que en el entrenamiento: suma de los píxeles válidos entre todos los píxeles
        pixel_loss = loss_function(masks, logits).numpy()
        loss_sum += float(np.sum(pixel_loss * weights.numpy()))
        pixel_count += weights.numpy().size

        # Solo los píxeles con peso 1 entran a la matriz de confusión
        valid = weights.numpy() > 0
        true_labels = masks.numpy()[valid]
        pred_labels = predicted[valid]

        # Cada par (clase real, clase predicha) se convierte en una posición de la matriz y se cuenta
        confusion += np.bincount(true_labels * NUM_CLASSES + pred_labels, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion, loss_sum / pixel_count


def metrics_from_confusion(confusion):
    ''' Calcula la exactitud por píxel, el IoU de cada clase y el mIoU a partir de la matriz de confusión '''

    true_positives = np.diag(confusion)
    false_positives = confusion.sum(axis=0) - true_positives
    false_negatives = confusion.sum(axis=1) - true_positives

    accuracy = true_positives.sum() / confusion.sum()

    # IoU = intersección entre unión; las clases que no aparecen ni se predicen se marcan como NaN y no cuentan en el promedio
    union = true_positives + false_positives + false_negatives
    iou = np.where(union > 0, true_positives / np.maximum(union, 1), np.nan)
    return accuracy, iou, np.nanmean(iou)


def plot_iou_vs_frequency(table, output_file):
    ''' Grafica el IoU de cada clase en prueba contra el porcentaje de píxeles que ocupa en entrenamiento '''

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.5,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.scatter(table["pct_pixeles_entrenamiento"], table["iou_prueba"], s=10, color="#2a78d6", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Porcentaje de píxeles en entrenamiento (escala logarítmica)")
    ax.set_ylabel("IoU en prueba")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ''' Evalúa el modelo indicado y guarda sus métricas en results/evaluacion/ '''

    # El nombre del experimento se pasa al correr el script, por ejemplo: python src/evaluate.py v1
    experiment = sys.argv[1] if len(sys.argv) > 1 else "v1"
    output_dir = RESULTS_DIR / "evaluacion" / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    # compile=False porque solo se necesita el modelo para predecir, no para seguir entrenando
    model = tf.keras.models.load_model(MODELS_DIR / f"{experiment}.keras", compile=False)
    _, val_batches, test_batches = get_datasets()

    names = load_class_names()
    summary, iou_columns = [], {}

    for name, dataset in [("validacion", val_batches), ("prueba", test_batches)]:
        print(f"Evaluando {name}...", flush=True)
        confusion, loss = confusion_and_loss(model, dataset)
        accuracy, iou, miou = metrics_from_confusion(confusion)

        summary.append({
            "conjunto": name,
            "loss": round(loss, 4),
            "exactitud": round(accuracy, 4),
            "miou": round(miou, 4),
            "clases_con_iou_0": int(np.sum(iou == 0)),
            "clases_con_iou_mayor_a_0.5": int(np.sum(iou > 0.5)),
        })
        iou_columns[f"iou_{name}"] = np.round(iou, 4)

    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / "metricas.csv", index=False, encoding="utf-8-sig")

    # La clase k del modelo corresponde a la clase k + 1 del dataset, porque la clase 0 se ignoró
    table = pd.DataFrame({"id_dataset": range(1, NUM_CLASSES + 1), "nombre": [names.get(c, str(c)) for c in range(1, NUM_CLASSES + 1)]})
    for column, values in iou_columns.items():
        table[column] = values

    # Si existe la exploración, se agrega la frecuencia de cada clase para comparar desbalance contra IoU
    if EXPLORATION_FILE.exists():
        frequency = pd.read_csv(EXPLORATION_FILE, encoding="utf-8-sig")[["id", "pct_pixeles_entrenamiento"]]
        table = table.merge(frequency, left_on="id_dataset", right_on="id", how="left").drop(columns="id")
        plot_iou_vs_frequency(table, output_dir / "iou_vs_frecuencia.png")

    table = table.sort_values("iou_prueba", ascending=False)
    table.to_csv(output_dir / "iou_por_clase.csv", index=False, encoding="utf-8-sig")

    print(f"\nExperimento {experiment}")
    print(summary.to_string(index=False))
    print("\nClases con mayor IoU en prueba:")
    print(table.head(10)[["nombre", "iou_validacion", "iou_prueba"]].to_string(index=False))
    print("\nClases con menor IoU en prueba:")
    print(table.tail(10)[["nombre", "iou_validacion", "iou_prueba"]].to_string(index=False))

    # Correlación de rangos: indica si las clases más frecuentes tienden a tener mejor IoU
    if "pct_pixeles_entrenamiento" in table:
        correlation = table["pct_pixeles_entrenamiento"].rank().corr(table["iou_prueba"].rank())
        print(f"\nCorrelación de rangos entre frecuencia de la clase e IoU en prueba: {correlation:.2f}")

    print(f"\nResultados guardados en {output_dir}")


if __name__ == "__main__":
    main()
