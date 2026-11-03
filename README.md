# Segmentación semántica de escenas con U-Net

Proyecto académico de la subcompetencia SMA0401C Aprendizaje e IA del Tecnológico de Monterrey.

El objetivo es entrenar un modelo de aprendizaje profundo que asigne a cada píxel de una imagen una de 150 clases, por ejemplo cielo, pared, árbol, persona, calle o carro. Se usa una arquitectura U-Net entrenada desde cero con TensorFlow y Keras, y se documentan los experimentos realizados para mejorarla.

## Dataset

Se usa ADE20K en la versión del Scene Parsing Benchmark, conocida como SceneParse150. Contiene 20,210 imágenes de entrenamiento y 2,000 de validación con sus máscaras, 150 clases y escenas de interiores y exteriores.

Página oficial: http://sceneparsing.csail.mit.edu/

El dataset no se incluye en este repositorio. Sus imágenes se publican solo para uso no comercial, de investigación y educativo. Los términos de uso están en https://github.com/CSAILVision/ADE20K#terms

### Descarga

Desde la raíz del repositorio. El archivo pesa alrededor de 1 GB.

```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -P data/
unzip data/ADEChallengeData2016.zip -d data/
```

### Separación de datos

Las anotaciones del set de prueba oficial no son públicas. Por eso el set de entrenamiento oficial se divide en entrenamiento y validación con una semilla fija, y las 2,000 imágenes de validación oficiales se usan como set de prueba.

## Requisitos

- Linux. El proyecto se desarrolla en Fedora 44.
- Python 3.13.
- GPU NVIDIA con el driver instalado. Es opcional, pero sin ella el entrenamiento es mucho más lento.

## Instalación

```bash
git clone https://github.com/taqueritospro/image-segmentation-deep-learning.git
cd image-segmentation-deep-learning
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Para comprobar que TensorFlow detecta la GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si no se cuenta con una GPU NVIDIA, se debe instalar `tensorflow==2.21.0` en lugar de `tensorflow[and-cuda]`.

## Problemas comunes

### TensorFlow no detecta la GPU por libcusolver.so.11

Si el comando anterior devuelve una lista vacía y aparece el aviso `Cannot dlopen some GPU libraries`, se puede ver qué librería falló con el registro detallado de TensorFlow:

```bash
TF_CPP_MAX_VLOG_LEVEL=3 python -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" 2>&1 | grep -i "could not load"
```

En este proyecto, con Fedora 44 y TensorFlow 2.21.0, la única librería que falló fue `libcusolver.so.11`. Pip sí la instala dentro de `.venv`, pero TensorFlow no la busca en esa carpeta. La solución es agregar las carpetas de librerías de NVIDIA a `LD_LIBRARY_PATH` cada vez que se activa el entorno:

```bash
cat >> .venv/bin/activate << 'EOF'

# Librerías de CUDA instaladas por pip: TensorFlow no encuentra libcusolver por sí solo
NVIDIA_LIBS=$(find "$VIRTUAL_ENV"/lib*/python3.13/site-packages/nvidia -maxdepth 2 -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
```

La ruta usa `lib*` porque algunas distribuciones crean el entorno en `lib` y otras, como Fedora, en `lib64`. Después hay que abrir una terminal nueva, activar el entorno y volver a comprobar la GPU.

## Estructura del repositorio

```
image-segmentation-deep-learning/
├── data/      # aquí se descarga ADE20K, no se sube
├── models/    # modelos .keras entrenados, no se suben
├── src/       # descarga, pipeline de datos, modelo, entrenamiento y evaluación
├── results/   # gráficas, métricas y tabla de experimentos
├── paper/     # paper del proyecto en PDF
└── app/       # interfaz para generar predicciones
```

## Uso

Se documentará cuando estén listos el entrenamiento, la evaluación y la interfaz.

## Cita del dataset

B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso y A. Torralba, "Scene Parsing through ADE20K Dataset", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. MIT CSAIL.

```bibtex
@inproceedings{zhou2017scene,
  title     = {Scene Parsing through ADE20K Dataset},
  author    = {Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017}
}
```

## Autor

Oswaldo Isaias Hernández Santes

## Licencia

El código se publica bajo la licencia MIT, ver el archivo LICENSE. El dataset ADE20K conserva su propia licencia y no forma parte de este repositorio.
