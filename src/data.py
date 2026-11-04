''' Pipeline de datos: divide ADE20K en entrenamiento, validación y prueba y prepara los batches para el modelo '''

import csv
import resource
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import ANNOTATIONS_DIR, BATCH_SIZE, CACHE, CACHE_DIR, IGNORE_LABEL, IMAGES_DIR, IMG_SIZE, NUM_CLASSES, RESULTS_DIR, SEED, SHUFFLE_BUFFER, VAL_FRACTION

SPLIT_FILE = RESULTS_DIR / "division_datos.csv"


def list_pairs(split):
    ''' Regresa la lista ordenada de pares (imagen, máscara) de un conjunto oficial del dataset '''

    images = {path.stem: path for path in (IMAGES_DIR / split).glob("*.jpg")}
    masks = {path.stem: path for path in (ANNOTATIONS_DIR / split).glob("*.png")}
    return [(str(images[name]), str(masks[name])) for name in sorted(images.keys() & masks.keys())]


def split_data():
    ''' Divide el entrenamiento oficial en entrenamiento y validación, y usa la validación oficial como prueba '''

    pairs = list_pairs("training")

    # La lista se revuelve con la semilla fija para que la división sea siempre la misma
    order = np.random.default_rng(SEED).permutation(len(pairs))
    val_count = round(len(pairs) * VAL_FRACTION)

    return {
        "entrenamiento": [pairs[i] for i in order[val_count:]],
        "validacion": [pairs[i] for i in order[:val_count]],
        "prueba": list_pairs("validation"),
    }


def save_split(splits):
    ''' Guarda en un CSV qué imagen quedó en cada conjunto, para que la división sea reproducible '''

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["conjunto", "imagen"])
        for name, pairs in splits.items():
            for image_path, _ in pairs:
                writer.writerow([name, Path(image_path).name])


def load_pair(image_path, mask_path):
    ''' Lee una imagen y su máscara y las redimensiona a IMG_SIZE x IMG_SIZE '''

    # Se piden 3 canales para que las imágenes en escala de grises queden como RGB
    image = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
    mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="bilinear")

    # Vecino más cercano para que ningún píxel quede con un valor que mezcle dos clases
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")

    # Se guardan como uint8 para que la caché ocupe 4 veces menos que en float32
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    mask = tf.cast(mask[..., 0], tf.uint8)
    return image, mask


def prepare(image, mask):
    ''' Normaliza la imagen y ajusta las etiquetas de la máscara para el modelo '''

    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)

    # La clase 0 del dataset se marca para ignorarse; las clases 1 a 150 pasan a 0 a 149
    mask = tf.where(mask == 0, IGNORE_LABEL, mask - 1)
    return image, mask


def make_dataset(pairs, training=False, cache_name=None):
    ''' Arma el pipeline de un conjunto: lectura, redimensión, caché, revoltura, normalización y batches '''

    image_paths = [image for image, _ in pairs]
    mask_paths = [mask for _, mask in pairs]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Las imágenes se leen y redimensionan en paralelo usando varios núcleos del CPU
    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)

    # La caché guarda las imágenes ya redimensionadas; desde la segunda época ya no se vuelven a leer del disco
    if cache_name and CACHE == "ram":
        dataset = dataset.cache()
    elif cache_name and CACHE == "disk":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset = dataset.cache(str(CACHE_DIR / cache_name))

    # Solo el entrenamiento se revuelve, y en cada época con un orden distinto
    if training:
        dataset = dataset.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)

    # Mientras la GPU entrena con un batch, el CPU ya prepara el siguiente
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_datasets():
    ''' Regresa los datasets de entrenamiento, validación y prueba listos para el modelo '''

    splits = split_data()
    save_split(splits)

    # La prueba no usa caché porque solo se evalúa una vez
    train = make_dataset(splits["entrenamiento"], training=True, cache_name="entrenamiento")
    val = make_dataset(splits["validacion"], cache_name="validacion")
    test = make_dataset(splits["prueba"])
    return train, val, test


def max_ram_gb():
    ''' Regresa la RAM máxima que ha usado este proceso, en GB '''

    # En Linux ru_maxrss viene en kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def main():
    ''' Revisa el pipeline: tamaños de los conjuntos, forma de los batches, etiquetas, velocidad y RAM usada '''

    splits = split_data()
    save_split(splits)
    for name, pairs in splits.items():
        print(f"{name}: {len(pairs)} imágenes")
    print(f"División guardada en {SPLIT_FILE}")

    train = make_dataset(splits["entrenamiento"], training=True, cache_name="entrenamiento")
    total_batches = int(np.ceil(len(splits["entrenamiento"]) / BATCH_SIZE))
    labels = set()
    count = 0

    # Primera época completa: lee y redimensiona cada imagen y llena la caché
    print(f"\nPrimera época, sin caché, caché configurada en: {CACHE}")
    start = time.perf_counter()
    for index, (images, masks) in enumerate(train, start=1):

        if index == 1:
            print(f"  Batch de imágenes: {images.shape}, {images.dtype}, valores entre {images.numpy().min():.2f} y {images.numpy().max():.2f}")
            print(f"  Batch de máscaras: {masks.shape}, {masks.dtype}")

        # Las etiquetas se revisan en los primeros 100 batches para no frenar la medición de velocidad
        if index <= 100:
            labels.update(np.unique(masks.numpy()).tolist())

        count += images.shape[0]

        if index % 500 == 0 or index == total_batches:
            print(f"  {index}/{total_batches} batches, RAM máxima {max_ram_gb():.1f} GB", flush=True)

    first_rate = count / (time.perf_counter() - start)

    # Segunda pasada parcial: las imágenes ya salen de la caché
    count = 0
    start = time.perf_counter()
    for images, _ in train.take(300):
        count += images.shape[0]
    second_rate = count / (time.perf_counter() - start)

    # Las etiquetas válidas son 0 a 149 y la etiqueta de ignorar
    invalid = sorted(label for label in labels if not (0 <= label < NUM_CLASSES or label == IGNORE_LABEL))

    print(f"\nEtiquetas encontradas: {len(labels)} valores distintos, de {min(labels)} a {max(labels)}")
    print(f"Etiquetas fuera de rango: {invalid if invalid else 'ninguna'}")
    print(f"Velocidad sin caché: {first_rate:.0f} imágenes por segundo")
    print(f"Velocidad con caché: {second_rate:.0f} imágenes por segundo")
    print(f"RAM máxima usada por el proceso: {max_ram_gb():.1f} GB")


if __name__ == "__main__":
    main()
