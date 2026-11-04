''' Explora ADE20K: verifica los archivos, mide las imágenes y calcula la distribución de clases '''

from collections import Counter

import matplotlib

# Backend sin ventana: las gráficas solo se guardan como archivo
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import ANNOTATIONS_DIR, CLASS_INFO_FILE, EXPECTED_COUNTS, IMAGES_DIR, IMG_SIZE, NUM_CLASSES, RESULTS_DIR, SPLITS

OUTPUT_DIR = RESULTS_DIR / "exploracion"
SPLIT_NAMES = {"training": "entrenamiento", "validation": "validacion"}

# Estilo de las gráficas para el paper: letra serif de 8 puntos como pide la plantilla IEEE
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": "#898781",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BAR_COLOR = "#2a78d6"

# Ancho de una columna y de la página completa de la plantilla IEEE, en pulgadas
COLUMN_WIDTH = 3.5
PAGE_WIDTH = 7.16


def load_class_names():
    ''' Lee los nombres de las 150 clases del archivo que trae el dataset '''

    names = {0: "sin etiqueta"}
    with open(CLASS_INFO_FILE, encoding="utf-8") as file:
        for line in file:

            # Las columnas vienen separadas por tabulador; si no, por espacios
            columns = line.rstrip("\n").split("\t") if "\t" in line else line.split(None, 4)

            # Se saltan el encabezado y cualquier línea que no empiece con el número de clase
            if not columns or not columns[0].strip().isdigit():
                continue

            # El nombre puede traer sinónimos separados por coma; se usa el primero
            names[int(columns[0])] = columns[-1].split(",")[0].strip()

    if len(names) != NUM_CLASSES + 1:
        print(f"Aviso: se leyeron {len(names) - 1} nombres de clase, se esperaban {NUM_CLASSES}.")

    return names


def match_files(split):
    ''' Empareja cada imagen con su máscara por nombre de archivo '''

    images = {path.stem: path for path in (IMAGES_DIR / split).glob("*.jpg")}
    masks = {path.stem: path for path in (ANNOTATIONS_DIR / split).glob("*.png")}
    common = sorted(images.keys() & masks.keys())

    return {
        "pairs": [(images[name], masks[name]) for name in common],
        "images": len(images),
        "masks": len(masks),
        "without_mask": len(images.keys() - masks.keys()),
        "without_image": len(masks.keys() - images.keys()),
    }


def analyze_split(pairs):
    ''' Mide las imágenes y cuenta los píxeles de cada clase en las máscaras '''

    widths, heights = [], []
    image_modes, mask_modes = Counter(), Counter()
    size_mismatch = 0
    out_of_range = 0
    pixel_counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    image_counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)

    for index, (image_path, mask_path) in enumerate(pairs, start=1):

        # Image.open solo lee el encabezado, así que obtener el tamaño y el modo de color es rápido
        with Image.open(image_path) as image:
            width, height = image.size
            image_modes[image.mode] += 1

        with Image.open(mask_path) as mask_file:
            mask_modes[mask_file.mode] += 1
            mask = np.array(mask_file)

        widths.append(width)
        heights.append(height)

        # La máscara debe medir lo mismo que la imagen para que cada píxel tenga su clase
        if mask.shape[:2] != (height, width):
            size_mismatch += 1

        # Una máscara con más de un canal no se puede contar como índices de clase
        if mask.ndim != 2:
            continue

        counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES + 1)

        # Valores mayores a 150 no corresponden a ninguna clase
        if len(counts) > NUM_CLASSES + 1:
            out_of_range += 1
            counts = counts[:NUM_CLASSES + 1]

        pixel_counts += counts
        image_counts += counts > 0

        if index % 2000 == 0 or index == len(pairs):
            print(f"  {index}/{len(pairs)} imágenes procesadas")

    return {
        "widths": np.array(widths),
        "heights": np.array(heights),
        "image_modes": image_modes,
        "mask_modes": mask_modes,
        "size_mismatch": size_mismatch,
        "out_of_range": out_of_range,
        "pixel_counts": pixel_counts,
        "image_counts": image_counts,
    }


def format_modes(modes):
    ''' Convierte el conteo de modos de color en texto, por ejemplo "RGB: 20000; L: 210" '''

    return "; ".join(f"{mode}: {count}" for mode, count in modes.most_common())


def save_summary(files, stats):
    ''' Guarda en un CSV el resumen de archivos y tamaños de cada conjunto '''

    metrics = [
        "imagenes_esperadas", "imagenes", "mascaras", "imagenes_sin_mascara", "mascaras_sin_imagen",
        "mascaras_de_tamano_distinto", "mascaras_con_valores_mayores_a_150",
        "ancho_min", "ancho_max", "ancho_media", "ancho_mediana",
        "alto_min", "alto_max", "alto_media", "alto_mediana",
        "modos_de_color_imagenes", "modos_de_color_mascaras",
    ]
    table = pd.DataFrame({"metrica": metrics})

    # Una columna por conjunto, con los valores en el mismo orden que la lista de métricas
    for s in SPLITS:
        widths, heights = stats[s]["widths"], stats[s]["heights"]
        table[SPLIT_NAMES[s]] = [
            EXPECTED_COUNTS[s], files[s]["images"], files[s]["masks"], files[s]["without_mask"], files[s]["without_image"],
            stats[s]["size_mismatch"], stats[s]["out_of_range"],
            widths.min(), widths.max(), round(widths.mean(), 1), np.median(widths),
            heights.min(), heights.max(), round(heights.mean(), 1), np.median(heights),
            format_modes(stats[s]["image_modes"]), format_modes(stats[s]["mask_modes"]),
        ]

    # utf-8-sig hace que Excel muestre bien los acentos al abrir el CSV
    table.to_csv(OUTPUT_DIR / "resumen.csv", index=False, encoding="utf-8-sig")


def class_percentages(split_stats):
    ''' Calcula el porcentaje de píxeles de cada clase y el porcentaje de imágenes donde aparece '''

    pixel_pct = split_stats["pixel_counts"] / split_stats["pixel_counts"].sum() * 100
    image_pct = split_stats["image_counts"] / len(split_stats["widths"]) * 100

    return pixel_pct, image_pct


def save_class_distribution(stats, names):
    ''' Guarda en un CSV la distribución de las 150 clases y de la clase 0 en cada conjunto '''

    percentages = {s: class_percentages(stats[s]) for s in SPLITS}

    class_ids = range(NUM_CLASSES + 1)
    table = pd.DataFrame({"id": class_ids, "nombre": [names.get(c, "") for c in class_ids]})
    for s in SPLITS:
        pixel_pct, image_pct = percentages[s]
        table[f"pct_pixeles_{SPLIT_NAMES[s]}"] = np.round(pixel_pct, 4)
        table[f"pct_imagenes_{SPLIT_NAMES[s]}"] = np.round(image_pct, 2)

    table.to_csv(OUTPUT_DIR / "distribucion_clases.csv", index=False, encoding="utf-8-sig")

    return percentages


def plot_class_frequency(pixel_pct):
    ''' Grafica el porcentaje de píxeles de las 150 clases ordenadas de más a menos frecuente '''

    # La clase 0 se excluye porque no es una clase que el modelo vaya a predecir
    values = np.sort(pixel_pct[1:])[::-1]
    positions = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    ax.bar(positions, values, width=0.8, color=BAR_COLOR)
    ax.set_yscale("log")
    ax.set_xlim(0, len(values) + 1)
    ax.set_xlabel("Clases ordenadas de más a menos frecuente")
    ax.set_ylabel("Porcentaje de píxeles (escala logarítmica)")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUTPUT_DIR / "frecuencia_clases.png")
    plt.close(fig)


def plot_extreme_classes(pixel_pct, names, count=15):
    ''' Grafica las clases más y menos frecuentes con su nombre '''

    order = np.argsort(pixel_pct[1:])[::-1] + 1
    groups = [("Más frecuentes", order[:count]), ("Menos frecuentes", order[-count:])]

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.8))
    for ax, (title, class_ids) in zip(axes, groups):

        # Se invierte el orden para que la clase con más píxeles quede arriba
        class_ids = class_ids[::-1]
        ax.barh([names.get(c, str(c)) for c in class_ids], pixel_pct[class_ids], height=0.7, color=BAR_COLOR)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Porcentaje de píxeles")
        ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clases_extremos.png")
    plt.close(fig)


def plot_image_sizes(widths, heights):
    ''' Grafica la distribución del ancho y el alto de las imágenes junto al tamaño de entrada del modelo '''

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.2), sharey=True)
    for ax, values, label in zip(axes, [widths, heights], ["Ancho (píxeles)", "Alto (píxeles)"]):
        ax.hist(values, bins=50, color=BAR_COLOR, edgecolor="white", linewidth=0.3)

        # Línea de referencia del tamaño al que se redimensionan las imágenes para el modelo
        ax.axvline(IMG_SIZE, color="#0b0b0b", linewidth=1, linestyle="--")
        ax.text(IMG_SIZE, ax.get_ylim()[1] * 0.95, f" Entrada del modelo ({IMG_SIZE})", va="top", fontsize=7, color="#52514e", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

        ax.set_xlabel(label)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Número de imágenes")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tamanos_imagenes.png")
    plt.close(fig)


def print_findings(files, stats, percentages, names):
    ''' Imprime en consola los hallazgos principales '''

    for s in SPLITS:
        pixel_pct, _ = percentages[s]
        ok = files[s]["images"] == EXPECTED_COUNTS[s] and files[s]["without_mask"] == 0 and files[s]["without_image"] == 0
        print(f"\nConjunto {SPLIT_NAMES[s]}")
        print(f"  Imágenes: {files[s]['images']} de {EXPECTED_COUNTS[s]} esperadas, archivos {'completos' if ok else 'INCOMPLETOS'}")
        print(f"  Máscaras de tamaño distinto a su imagen: {stats[s]['size_mismatch']}")
        print(f"  Máscaras con valores mayores a 150: {stats[s]['out_of_range']}")
        print(f"  Píxeles sin etiqueta, clase 0: {pixel_pct[0]:.2f}%")

    # Las clases extremas se reportan con el conjunto de entrenamiento, que es del que aprende el modelo
    pixel_pct, image_pct = percentages["training"]
    order = np.argsort(pixel_pct[1:])[::-1] + 1
    print("\nClases con más píxeles en entrenamiento:")
    for c in order[:5]:
        print(f"  {names.get(c, c)}: {pixel_pct[c]:.2f}% de píxeles, aparece en {image_pct[c]:.1f}% de imágenes")
    print("Clases con menos píxeles en entrenamiento:")
    for c in order[-5:]:
        print(f"  {names.get(c, c)}: {pixel_pct[c]:.4f}% de píxeles, aparece en {image_pct[c]:.1f}% de imágenes")

    # Una clase sin píxeles no se puede usar para calcular la proporción, así que se reporta aparte
    missing = [names.get(c, c) for c in range(1, NUM_CLASSES + 1) if pixel_pct[c] == 0]
    if missing:
        print(f"Clases sin ningún píxel en entrenamiento: {', '.join(map(str, missing))}")

    present = order[pixel_pct[order] > 0]
    ratio = pixel_pct[present[0]] / pixel_pct[present[-1]]
    print(f"La clase más frecuente tiene {ratio:,.0f} veces más píxeles que la menos frecuente con píxeles.")


def main():
    ''' Ejecuta la exploración completa y guarda los resultados en results/exploracion/ '''

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = load_class_names()

    files, stats = {}, {}
    for split in SPLITS:
        print(f"Analizando {SPLIT_NAMES[split]}...")
        files[split] = match_files(split)
        stats[split] = analyze_split(files[split]["pairs"])

    save_summary(files, stats)
    percentages = save_class_distribution(stats, names)

    # Las gráficas usan el conjunto de entrenamiento
    plot_class_frequency(percentages["training"][0])
    plot_extreme_classes(percentages["training"][0], names)
    plot_image_sizes(stats["training"]["widths"], stats["training"]["heights"])

    print_findings(files, stats, percentages, names)
    print(f"\nResultados guardados en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
