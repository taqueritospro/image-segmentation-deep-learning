''' Entrena la U-Net desde cero y guarda el modelo, su historial, sus curvas y su fila en la tabla de experimentos '''

import time

import matplotlib

# Backend sin ventana: las gráficas solo se guardan como archivo
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from config import BATCH_SIZE, IMG_SIZE, MODELS_DIR, NUM_CLASSES, RESULTS_DIR, SEED
from data import get_datasets

# Datos del experimento: para uno nuevo se cambian estas líneas
EXPERIMENT = "v2"
DESCRIPTION = "v1 con data augmentation: volteo horizontal aleatorio en entrenamiento"
EPOCHS = 50
LEARNING_RATE = 0.001
BASE_FILTERS = 32
DEPTH = 4
DROPOUT = False
AUGMENT = True

EXPERIMENTS_DIR = RESULTS_DIR / "experimentos"
TABLE_FILE = EXPERIMENTS_DIR / "tabla_experimentos.csv"


def conv_block(filters):
    ''' Dos Conv2D 3x3, cada una con batch normalization y ReLU '''

    block = tf.keras.Sequential()
    for _ in range(2):

        # La convolución no lleva sesgo porque batch normalization ya agrega su propio desplazamiento
        block.add(tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False))
        block.add(tf.keras.layers.BatchNormalization())
        block.add(tf.keras.layers.ReLU())

    return block


def upsample(filters, size, apply_dropout=False):
    ''' Duplica el tamaño de la imagen con Conv2DTranspose, igual que pix2pix.upsample del notebook de clase '''

    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same", use_bias=False))
    block.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        block.add(tf.keras.layers.Dropout(0.5))

    block.add(tf.keras.layers.ReLU())
    return block


def unet_model(output_channels):
    ''' U-Net con codificador de bloques convolucionales y decodificador con skip connections '''

    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    # En cada nivel del codificador se duplican los filtros: 32, 64, 128, 256
    down_stack = [conv_block(BASE_FILTERS * 2 ** level) for level in range(DEPTH)]

    # El decodificador recorre los niveles al revés: 256, 128, 64, 32
    up_stack = [upsample(BASE_FILTERS * 2 ** level, 3, DROPOUT) for level in reversed(range(DEPTH))]
    merge_stack = [conv_block(BASE_FILTERS * 2 ** level) for level in reversed(range(DEPTH))]

    # Codificador: se guarda la salida de cada nivel para las skip connections y se reduce la imagen a la mitad
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    # Nivel más profundo: 16x16 con 512 filtros
    x = conv_block(BASE_FILTERS * 2 ** DEPTH)(x)

    # Decodificador: se duplica el tamaño, se concatena con el nivel correspondiente del codificador y se combinan con dos Conv2D
    for up, merge, skip in zip(up_stack, merge_stack, reversed(skips)):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = merge(x)

    # Una salida por clase en cada píxel; se deja en float32 aunque se use precisión mixta, para que la pérdida sea estable
    outputs = tf.keras.layers.Conv2D(output_channels, 1, dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def plot_curves(history, output_file):
    ''' Grafica la loss y el mIoU de entrenamiento y validación por época, con el formato de las gráficas del paper '''

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

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    for ax, metric, label in zip(axes, ["loss", "miou"], ["Loss (entropía cruzada)", "mIoU"]):

        # Entrenamiento y validación se distinguen por color y por tipo de línea, para que también se lean impresas en blanco y negro
        ax.plot(history.index, history[metric], color="#2a78d6", linestyle="-", linewidth=1.5, label="Entrenamiento")
        ax.plot(history.index, history["val_" + metric], color="#eb6834", linestyle="--", linewidth=1.5, label="Validación")

        ax.set_xlabel("Época")
        ax.set_ylabel(label)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table_row(history, minutes):
    ''' Agrega a la tabla de experimentos los resultados de la mejor época según el mIoU de validación '''

    best = history["val_miou"].idxmax()
    row = pd.DataFrame([{
        "experimento": EXPERIMENT,
        "descripcion": DESCRIPTION,
        "epocas": EPOCHS,
        "tasa_aprendizaje": LEARNING_RATE,
        "filtros_iniciales": BASE_FILTERS,
        "niveles": DEPTH,
        "dropout": DROPOUT,
        "augmentation": AUGMENT,
        "batch": BATCH_SIZE,
        "mejor_epoca": best,
        "loss_entrenamiento": round(history.loc[best, "loss"], 4),
        "loss_validacion": round(history.loc[best, "val_loss"], 4),
        "exactitud_entrenamiento": round(history.loc[best, "exactitud"], 4),
        "exactitud_validacion": round(history.loc[best, "val_exactitud"], 4),
        "miou_entrenamiento": round(history.loc[best, "miou"], 4),
        "miou_validacion": round(history.loc[best, "val_miou"], 4),
        "minutos": round(minutes, 1),
    }])

    # Si la tabla ya existe, la fila nueva se agrega al final
    if TABLE_FILE.exists():
        row = pd.concat([pd.read_csv(TABLE_FILE, encoding="utf-8-sig"), row], ignore_index=True)

    row.to_csv(TABLE_FILE, index=False, encoding="utf-8-sig")
    return best


def main():
    ''' Entrena el experimento, guarda el mejor modelo según el mIoU de validación y registra sus resultados '''

    output_dir = EXPERIMENTS_DIR / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Semilla fija para que los pesos iniciales y el orden de los datos sean reproducibles
    tf.keras.utils.set_random_seed(SEED)

    # La precisión mixta calcula en 16 bits donde es seguro, lo que reduce memoria y casi duplica la velocidad
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_batches, val_batches, _ = get_datasets(augment=AUGMENT)

    model = unet_model(output_channels=NUM_CLASSES)

    # Las métricas van en weighted_metrics para que usen los pesos por píxel y no cuenten la clase 0
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="exactitud"),
            tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_pred=False, name="miou"),
        ],
    )
    print(f"Experimento {EXPERIMENT}: {model.count_params():,} parámetros")

    callbacks = [

        # Solo se guarda el modelo cuando mejora el mIoU de validación
        tf.keras.callbacks.ModelCheckpoint(str(MODELS_DIR / f"{EXPERIMENT}.keras"), monitor="val_miou", mode="max", save_best_only=True, verbose=1),

        # El historial se escribe al terminar cada época, así no se pierde si el entrenamiento se interrumpe
        tf.keras.callbacks.CSVLogger(str(output_dir / "historial.csv")),
    ]

    # verbose=2 imprime una línea por época, más legible que la barra de progreso al guardar la salida en un archivo
    start = time.perf_counter()
    model_history = model.fit(train_batches, validation_data=val_batches, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    minutes = (time.perf_counter() - start) / 60

    # El historial pasa a un DataFrame, como en el notebook de Fashion MNIST, con las épocas numeradas desde 1
    history = pd.DataFrame(model_history.history)
    history.index = history.index + 1

    plot_curves(history, output_dir / "curvas.png")
    best = save_table_row(history, minutes)

    print(f"\nMejor época: {best}, mIoU de validación {history.loc[best, 'val_miou']:.4f}")
    print(f"Modelo guardado en {MODELS_DIR / f'{EXPERIMENT}.keras'}")
    print(f"Historial y curvas en {output_dir}")
    print(f"Fila agregada a {TABLE_FILE}")


if __name__ == "__main__":
    main()
