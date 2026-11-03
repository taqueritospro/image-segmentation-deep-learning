# Segmentación semántica de escenas con U-Net

Proyecto académico de la subcompetencia **SMA0401C – Aprendizaje e IA** (Tecnológico de Monterrey).

El objetivo es entrenar un modelo de aprendizaje profundo que asigne a **cada píxel** de una imagen una de **150 clases** (cielo, pared, árbol, persona, calle, carro, etc.). Se usa una arquitectura **U-Net entrenada desde cero** con TensorFlow/Keras y se documentan los experimentos para mejorarla.

> **Estado:** en desarrollo.

## Dataset

Se usa **ADE20K** en la versión del Scene Parsing Benchmark (SceneParse150):

- 20,210 imágenes de entrenamiento y 2,000 de validación, con sus máscaras.
- 150 clases, escenas de interiores y exteriores.
- Página oficial: http://sceneparsing.csail.mit.edu/

**El dataset no se incluye en este repositorio.** Sus imágenes se publican solo para uso no comercial, de investigación y educativo. Términos: https://github.com/CSAILVision/ADE20K#terms

### Descarga

Desde la raíz del repositorio (~1 GB):

```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -P data/
unzip data/ADEChallengeData2016.zip -d data/
```

### Separación de datos

Las anotaciones del set de prueba oficial no son públicas. Por eso:

- **Entrenamiento y validación:** se separan del set de entrenamiento oficial con una semilla fija.
- **Prueba:** las 2,000 imágenes de validación oficiales.

## Requisitos

- Linux (probado en Fedora 44)
- Python 3.13 (TensorFlow 2.21 no soporta oficialmente Python 3.14)
- GPU NVIDIA con driver instalado (opcional, pero recomendada)

## Instalación

```bash
git clone https://github.com/<usuario>/image-segmentation-deep-learning.git
cd image-segmentation-deep-learning
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Comprobar que TensorFlow detecta la GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Sin GPU NVIDIA, instalar `tensorflow==2.21.0` en lugar de `tensorflow[and-cuda]`.

## Estructura del repositorio

```
image-segmentation-deep-learning/
├── data/      # ADE20K se descarga aquí (no se sube)
├── models/    # modelos .keras entrenados (no se suben)
├── src/       # descarga, pipeline de datos, modelo, entrenamiento y evaluación
├── results/   # gráficas, métricas y tabla de experimentos
├── paper/     # paper del proyecto (Word y PDF)
└── app/       # interfaz para generar predicciones
```

## Uso

Pendiente: se documentará al implementar el entrenamiento, la evaluación y la interfaz.

## Cita del dataset

B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso y A. Torralba, "Scene Parsing through ADE20K Dataset", *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017. MIT CSAIL.

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

El código se publica bajo la licencia MIT (ver [LICENSE](LICENSE)). El dataset ADE20K conserva su propia licencia y no forma parte de este repositorio.
