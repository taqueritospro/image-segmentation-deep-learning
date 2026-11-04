''' Pipeline de datos: divide ADE20K en entrenamiento, validación y prueba y prepara los batches para el modelo '''

import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import ANNOTATIONS_DIR, BATCH_SIZE, CACHE, CACHE_DIR, IMAGES_DIR, IMG_SIZE, NUM_CLASSES, RESULTS_DIR, SEED, SHUFFLE_BUFFER, VAL_FRACTION

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

    rows = [{"conjunto": name, "imagen": Path(image).name} for name, pairs in splits.items() for image, _ in pairs]
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SPLIT_FILE, index=False, encoding="utf-8-sig")


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
    ''' Normaliza la imagen, ajusta las etiquetas y crea el peso de cada píxel '''

    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)

    # Peso 0 para los píxeles de la clase 0 del dataset, así no cuentan en la pérdida ni en las métricas
    weights = tf.cast(mask > 0, tf.float32)

    # Las clases 1 a 150 pasan a 0 a 149; los píxeles de la clase 0 quedan en 0, pero con peso 0
    mask = tf.maximum(mask - 1, 0)
    return image, mask, weights


def random_flip(image, mask, weights):
    ''' Voltea horizontalmente la imagen, su máscara y sus pesos en la mitad de los casos '''

    # Una sola decisión aleatoria para los tres, así la máscara sigue correspondiendo a la imagen
    flip = tf.random.uniform(()) > 0.5

    # El eje 1 es el ancho en los tres tensores, así que voltearlo es el efecto espejo
    image = tf.where(flip, tf.reverse(image, axis=[1]), image)
    mask = tf.where(flip, tf.reverse(mask, axis=[1]), mask)
    weights = tf.where(flip, tf.reverse(weights, axis=[1]), weights)
    return image, mask, weights


def make_dataset(pairs, training=False, cache_name=None, augment=False):
    ''' Arma el pipeline de un conjunto: lectura, redimensión, caché, revoltura, normalización, data augmentation y batches '''

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

    # El volteo va después de la caché para que en cada época salga distinto
    if augment:
        dataset = dataset.map(random_flip, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE)

    # Mientras la GPU entrena con un batch, el CPU ya prepara el siguiente
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_datasets(augment=False):
    ''' Regresa los datasets de entrenamiento, validación y prueba; el data augmentation solo se aplica al entrenamiento '''

    splits = split_data()
    save_split(splits)

    # La prueba no usa caché porque solo se evalúa una vez
    train = make_dataset(splits["entrenamiento"], training=True, cache_name="entrenamiento", augment=augment)
    val = make_dataset(splits["validacion"], cache_name="validacion")
    test = make_dataset(splits["prueba"])
    return train, val, test


def max_ram_gb():
    ''' Regresa la RAM máxima que ha usado este proceso, en GB '''

    # En Linux ru_maxrss viene en kilobytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def main():
    ''' Revisa el pipeline: tamaños de los conjuntos, forma de los batches, etiquetas, pesos, velocidad y RAM usada '''

    splits = split_data()
    save_split(splits)
    for name, pairs in splits.items():
        print(f"{name}: {len(pairs)} imágenes")
    print(f"División guardada en {SPLIT_FILE}")

    train = make_dataset(splits["entrenamiento"], training=True, cache_name="entrenamiento")
    total_batches = int(np.ceil(len(splits["entrenamiento"]) / BATCH_SIZE))
    labels, weight_values = set(), set()
    zero_weight, total_pixels = 0, 0
    count = 0

    # Primera época completa: lee y redimensiona cada imagen y llena la caché
    print(f"\nPrimera época, sin caché, caché configurada en: {CACHE}")
    start = time.perf_counter()
    for index, (images, masks, weights) in enumerate(train, start=1):

        if index == 1:
            print(f"  Batch de imágenes: {images.shape}, {images.dtype}, valores entre {images.numpy().min():.2f} y {images.numpy().max():.2f}")
            print(f"  Batch de máscaras: {masks.shape}, {masks.dtype}")
            print(f"  Batch de pesos: {weights.shape}, {weights.dtype}")

        # Etiquetas y pesos se revisan en los primeros 100 batches para no frenar la medición de velocidad
        if index <= 100:
            labels.update(np.unique(masks.numpy()).tolist())
            weight_values.update(np.unique(weights.numpy()).tolist())
            zero_weight += int(np.sum(weights.numpy() == 0))
            total_pixels += weights.numpy().size

        count += images.shape[0]

        if index % 500 == 0 or index == total_batches:
            print(f"  {index}/{total_batches} batches, RAM máxima {max_ram_gb():.1f} GB", flush=True)

    first_rate = count / (time.perf_counter() - start)

    # Segunda pasada parcial: las imágenes ya salen de la caché
    count = 0
    start = time.perf_counter()
    for images, _, _ in train.take(300):
        count += images.shape[0]
    second_rate = count / (time.perf_counter() - start)

    invalid = sorted(label for label in labels if not 0 <= label < NUM_CLASSES)

    print(f"\nEtiquetas encontradas: {len(labels)} valores distintos, de {min(labels)} a {max(labels)}")
    print(f"Etiquetas fuera de rango: {invalid if invalid else 'ninguna'}")
    print(f"Valores de los pesos: {sorted(weight_values)}")
    print(f"Píxeles con peso 0, la clase 0 del dataset: {zero_weight / total_pixels * 100:.2f}%")
    print(f"Velocidad sin caché: {first_rate:.0f} imágenes por segundo")
    print(f"Velocidad con caché: {second_rate:.0f} imágenes por segundo")
    print(f"RAM máxima usada por el proceso: {max_ram_gb():.1f} GB")


if __name__ == "__main__":
    main()
