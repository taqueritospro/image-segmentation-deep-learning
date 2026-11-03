''' Descarga ADE20K (SceneParse150) y lo deja en data/ con la estructura que usan los demás scripts '''

import shutil
import sys
import urllib.request
import zipfile

from config import ANNOTATIONS_DIR, CLASS_INFO_FILE, DATA_DIR, DATASET_URL, EXPECTED_COUNTS, IMAGES_DIR

ZIP_PATH = DATA_DIR / "ADEChallengeData2016.zip"

# Carpeta raíz que trae el zip; su contenido se mueve directo a data/
ZIP_ROOT = DATA_DIR / "ADEChallengeData2016"


def dataset_ready():
    ''' Indica si el dataset ya está descomprimido en data/ '''

    return IMAGES_DIR.is_dir() and ANNOTATIONS_DIR.is_dir() and CLASS_INFO_FILE.is_file()


def show_progress(blocks, block_size, total_size):
    ''' Muestra en consola el porcentaje descargado '''

    # Si el servidor no envía el tamaño total no se puede calcular el porcentaje
    if total_size <= 0:
        return

    percent = min(blocks * block_size / total_size * 100, 100)
    print(f"\rDescargando: {percent:5.1f}%", end="", flush=True)


def download_zip():
    ''' Descarga el zip del dataset si todavía no existe '''

    if ZIP_PATH.is_file():
        print(f"El zip ya existe en {ZIP_PATH}, no se descarga de nuevo.")
        return

    # Se descarga a un archivo temporal para no dejar un zip incompleto si la descarga se interrumpe
    temp_path = ZIP_PATH.with_suffix(".zip.part")
    print(f"Descargando {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, temp_path, show_progress)
    print()
    temp_path.rename(ZIP_PATH)


def extract_zip():
    ''' Descomprime el zip y mueve su contenido directamente a data/ '''

    print("Descomprimiendo...")
    with zipfile.ZipFile(ZIP_PATH) as zip_file:
        zip_file.extractall(DATA_DIR)

    if not ZIP_ROOT.is_dir():
        sys.exit(f"No se encontró la carpeta {ZIP_ROOT} después de descomprimir.")

    for item in ZIP_ROOT.iterdir():
        target = DATA_DIR / item.name

        # Si ya existe algo con el mismo nombre se detiene para no mezclar archivos
        if target.exists():
            sys.exit(f"{target} ya existe. Revisa data/ antes de volver a correr el script.")

        shutil.move(item, target)

    ZIP_ROOT.rmdir()


def verify_counts():
    ''' Revisa que cada conjunto tenga el número esperado de imágenes y máscaras '''

    all_ok = True
    for split, expected in EXPECTED_COUNTS.items():
        images = len(list((IMAGES_DIR / split).glob("*.jpg")))
        masks = len(list((ANNOTATIONS_DIR / split).glob("*.png")))
        status = "correcto" if images == expected and masks == expected else "NO COINCIDE"
        print(f"{split}: {images} imágenes, {masks} máscaras, se esperaban {expected} ({status})")

        if status != "correcto":
            all_ok = False

    return all_ok


def main():
    ''' Descarga, descomprime y verifica el dataset '''

    DATA_DIR.mkdir(exist_ok=True)

    if dataset_ready():
        print("El dataset ya está en data/, no se descarga de nuevo.")
    else:
        download_zip()
        extract_zip()

        # El zip se borra para liberar alrededor de 1 GB de disco
        ZIP_PATH.unlink()

    if not verify_counts():
        sys.exit("El número de archivos no coincide con lo esperado.")

    print("Dataset listo.")


if __name__ == "__main__":
    main()
