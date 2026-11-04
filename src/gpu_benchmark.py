''' Mide cuánta memoria de la GPU y cuánto tiempo por paso usa una U-Net provisional con distintos tamaños de batch '''

import argparse
import csv
import json
import os
import subprocess
import sys
import time

from config import IMG_SIZE, NUM_CLASSES, RESULTS_DIR, SEED

OUTPUT_DIR = RESULTS_DIR / "gpu"

# Combinaciones que se prueban: precisión numérica, filtros iniciales de la U-Net y tamaño de batch
PRECISIONS = ["float32", "mixed_float16"]
BASE_FILTERS = [32, 64]
BATCH_SIZES = [2, 4, 8, 16]

# Los primeros pasos incluyen la compilación del modelo, así que no se toman en cuenta para el tiempo
WARMUP_STEPS = 3
MEASURED_STEPS = 10

# Tiempo máximo para una combinación, en segundos, por si la compilación se queda atorada
CHILD_TIMEOUT = 900

# Textos con los que TensorFlow reporta falta de memoria; XLA a veces la reporta como un fallo de autoajuste
OOM_MARKERS = ["out of memory", "resourceexhausted", "autotuning failed", "no valid config found"]

FIELDS = ["precision", "filtros_iniciales", "parametros", "batch", "estado", "memoria_max_mb", "ms_por_paso", "imagenes_por_segundo"]


def parse_args():
    ''' Lee las opciones de memoria desde la línea de comandos '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-limit", type=int, default=0, help="límite de memoria de la GPU en MB; 0 usa crecimiento gradual")
    parser.add_argument("--allocator", choices=["bfc", "async"], default="bfc", help="administrador de memoria de TensorFlow")

    # Opción interna que usa el proceso hijo para probar una sola combinación
    parser.add_argument("--child", nargs=3, metavar=("PRECISION", "FILTROS", "BATCH"), help=argparse.SUPPRESS)
    return parser.parse_args()


def is_out_of_memory(text):
    ''' Indica si un mensaje de error se debe a falta de memoria en la GPU '''

    text = text.lower()
    return any(marker in text for marker in OOM_MARKERS)


def run_child(precision, base_filters, batch_size, memory_limit):
    ''' Entrena unos pasos con datos aleatorios dentro del proceso hijo y regresa la memoria máxima y el tiempo por paso '''

    import numpy as np
    import tensorflow as tf
    import keras
    from keras import layers

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("TensorFlow no detecta la GPU. La prueba solo tiene sentido con GPU.")

    # Con un límite explícito se pide esa cantidad de memoria; con el administrador async no se configura nada más
    if memory_limit:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
    elif os.environ.get("TF_GPU_ALLOCATOR") != "cuda_malloc_async":
        tf.config.experimental.set_memory_growth(gpus[0], True)

    keras.mixed_precision.set_global_policy(precision)

    def conv_block(x, filters):
        ''' Dos convoluciones 3x3 con ReLU, el bloque básico de la U-Net '''

        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    skips = []
    x = inputs

    # Codificador: en cada nivel se duplican los filtros y se reduce la imagen a la mitad
    for level in range(4):
        x = conv_block(x, base_filters * 2 ** level)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

    x = conv_block(x, base_filters * 16)

    # Decodificador: se recupera el tamaño y se concatena con el nivel correspondiente del codificador
    for level in reversed(range(4)):
        x = layers.Conv2DTranspose(base_filters * 2 ** level, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[level]])
        x = conv_block(x, base_filters * 2 ** level)

    # La salida se deja en float32 aunque se use precisión mixta, para que la pérdida sea estable
    outputs = layers.Conv2D(NUM_CLASSES, 1, dtype="float32")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Datos aleatorios con la misma forma que los reales: imágenes RGB y máscaras con índices de clase
    rng = np.random.default_rng(SEED)
    images = rng.random((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    masks = rng.integers(0, NUM_CLASSES, (batch_size, IMG_SIZE, IMG_SIZE), dtype=np.int32)

    result = {"precision": precision, "filtros_iniciales": base_filters, "parametros": model.count_params(), "batch": batch_size}
    try:
        for _ in range(WARMUP_STEPS):
            model.train_on_batch(images, masks)

        start = time.perf_counter()
        for _ in range(MEASURED_STEPS):
            model.train_on_batch(images, masks)
        elapsed = time.perf_counter() - start

    except Exception as error:
        if not is_out_of_memory(str(error)):
            raise
        result.update({"estado": "sin memoria", "memoria_max_mb": "", "ms_por_paso": "", "imagenes_por_segundo": ""})
        return result

    # Algunos administradores de memoria no reportan el máximo usado
    try:
        peak_mb = round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024 ** 2)
    except Exception:
        peak_mb = ""

    ms_per_step = elapsed / MEASURED_STEPS * 1000
    result.update({
        "estado": "ok",
        "memoria_max_mb": peak_mb,
        "ms_por_paso": round(ms_per_step, 1),
        "imagenes_por_segundo": round(batch_size / (ms_per_step / 1000), 1),
    })
    return result


def launch(precision, base_filters, batch_size, args):
    ''' Prueba una combinación en un proceso separado, para que una falla no detenga las demás '''

    env = os.environ.copy()

    # Se ocultan los mensajes informativos y el volcado de memoria de TensorFlow
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if args.allocator == "async":
        env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    command = [sys.executable, __file__, "--memory-limit", str(args.memory_limit), "--allocator", args.allocator, "--child", precision, str(base_filters), str(batch_size)]
    base = {"precision": precision, "filtros_iniciales": base_filters, "parametros": "", "batch": batch_size, "memoria_max_mb": "", "ms_por_paso": "", "imagenes_por_segundo": ""}

    try:
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {**base, "estado": "tiempo agotado"}

    # El resultado se busca aunque el proceso haya terminado con error al cerrarse
    for line in process.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])

    if is_out_of_memory(process.stderr):
        return {**base, "estado": "sin memoria"}

    error_lines = process.stderr.strip().splitlines()
    print("  Error: " + (error_lines[-1] if error_lines else f"código {process.returncode}"), flush=True)
    return {**base, "estado": "error"}


def main():
    ''' Prueba todas las combinaciones y guarda cada resultado en results/gpu/ en cuanto termina '''

    args = parse_args()

    # Modo interno: el proceso hijo prueba una combinación e imprime su resultado
    if args.child:
        precision, base_filters, batch_size = args.child[0], int(args.child[1]), int(args.child[2])
        result = run_child(precision, base_filters, batch_size, args.memory_limit)
        print("RESULT " + json.dumps(result), flush=True)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # El nombre del archivo indica la configuración de memoria usada
    suffix = f"_limite_{args.memory_limit}mb" if args.memory_limit else ""
    suffix += "_async" if args.allocator == "async" else ""
    output_file = OUTPUT_DIR / f"benchmark{suffix}.csv"

    with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for precision in PRECISIONS:
            for base_filters in BASE_FILTERS:
                for batch_size in BATCH_SIZES:
                    print(f"Probando {precision}, {base_filters} filtros, batch {batch_size}...", flush=True)
                    result = launch(precision, base_filters, batch_size, args)

                    # Cada fila se escribe en cuanto termina para no perder resultados si algo falla
                    writer.writerow(result)
                    file.flush()

                    # Si un batch no cabe, los más grandes tampoco, así que se pasa a la siguiente combinación
                    if result["estado"] != "ok":
                        print(f"  {result['estado']}", flush=True)
                        break

                    print(f"  ok, memoria máxima {result['memoria_max_mb']} MB, {result['ms_por_paso']} ms por paso", flush=True)

    print(f"\nResultados guardados en {output_file}")


if __name__ == "__main__":
    main()
