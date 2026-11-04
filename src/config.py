''' Rutas y constantes compartidas por los scripts del proyecto '''

from pathlib import Path

# Raíz del repositorio, calculada desde la ubicación de este archivo para que los scripts funcionen desde cualquier carpeta
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Estructura de ADE20K (SceneParse150) dentro de data/
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CLASS_INFO_FILE = DATA_DIR / "objectInfo150.txt"

# Conjuntos del dataset y número de imágenes que debe tener cada uno
SPLITS = ["training", "validation"]
EXPECTED_COUNTS = {"training": 20210, "validation": 2000}

DATASET_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

# Parámetros del modelo
IMG_SIZE = 256
NUM_CLASSES = 150
SEED = 42

# Parámetros del pipeline de datos
BATCH_SIZE = 8
VAL_FRACTION = 0.1
SHUFFLE_BUFFER = 1000

# Dónde se guarda la caché de las imágenes ya redimensionadas: "ram", "disk" o None para no usar caché
CACHE = "ram"
CACHE_DIR = DATA_DIR / "cache"
