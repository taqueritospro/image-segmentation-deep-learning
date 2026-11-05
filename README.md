# Segmentación semántica de escenas con U-Net

Proyecto de Deep Learning 

Subcompetencia SMA0401C Aprendizaje e IA

El objetivo es entrenar un modelo de aprendizaje profundo que asigne a cada píxel de una imagen una de 150 clases, por ejemplo cielo, pared, árbol, persona, calle o carro. Se usa una arquitectura U-Net entrenada desde cero usando los frameworks TensorFlow y Keras y se documentan los métodos para mejorarla en un paper.

## Dataset

Se usa ADE20K en la versión del Scene Parsing Benchmark, conocida como SceneParse150. Contiene 20,210 imágenes de entrenamiento y 2,000 de validación con sus máscaras, 150 clases y escenas de interiores y exteriores.

Página oficial: http://sceneparsing.csail.mit.edu/

El dataset no se incluye en este repositorio. Sus imágenes se publican solo para uso no comercial, de investigación y educativo. Los términos de uso están en https://github.com/CSAILVision/ADE20K#terms

### Descarga

Desde la raíz del repositorio y con el entorno activo:

```bash
python src/download_data.py
```

El script descarga el zip oficial, que pesa alrededor de 1 GB, después lo descomprime en `data/`, borra el zip y verifica que estén las 20,210 imágenes de entrenamiento con las 2,000 de validación con máscaras, en caso de que el dataset ya está en `data/` solo hace la verificación.

### Exploración

```bash
python src/explore_data.py
```

Verifica que cada imagen tenga su máscara del mismo tamaño, mide las imágenes y calcula la distribución de las clases. Los resultados se guardan en `results/exploracion/`.

### Separación de datos

Las anotaciones del set de prueba oficial no son públicas, por eso el set de entrenamiento oficial se divide en entrenamiento y validación con una semilla fija y las 2,000 imágenes de validación oficiales se usan como set de prueba.

## Requisitos

- Linux. El proyecto se desarrolla en Fedora 45 (anteriormente Fedora 44).
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

Si el comando anterior devuelve una lista vacía y aparece el aviso `Cannot dlopen some GPU libraries` se puede ver qué librería falló con el registro detallado de TensorFlow:

```bash
TF_CPP_MAX_VLOG_LEVEL=3 python -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')" 2>&1 | grep -i "could not load"
```

En este proyecto al momento de probar esto se hizo con Fedora 44 y TensorFlow 2.21.0, la única librería que falló fue `libcusolver.so.11`. Pip sí la instala dentro de `.venv`, pero TensorFlow no la busca en esa carpeta. La solución es agregar las carpetas de librerías de NVIDIA a `LD_LIBRARY_PATH` cada vez que se activa el entorno:

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

Todos los comandos se corren desde la raíz del repositorio y con el entorno activo.

### Entrenamiento

```bash
python src/train.py
```

El nombre del experimento y sus hiperparámetros están como constantes al inicio de `src/train.py`, para un experimento nuevo se cambian esas líneas, el script guarda el mejor modelo según el mIoU de validación en `models/<experimento>.keras`, el historial y las curvas de loss y mIoU en `results/experimentos/<experimento>/` y agrega una fila a `results/experimentos/tabla_experimentos.csv`.

Cada entrenamiento de 50 épocas tarda alrededor de 4 horas con una RTX 3050 de 4 GB.

### Evaluación

```bash
python src/evaluate.py v2
```

Evalúa el modelo indicado en validación y en prueba, calcula la loss, la exactitud por píxel, el mIoU y el IoU de cada clase y lo compara con la frecuencia de cada clase en entrenamiento, los resultados se guardan en `results/evaluacion/<experimento>/`.

### Predicciones

```bash
# Puede hacerse en una foto directamente
python src/predict.py v2 ruta/a/foto.jpg

# O en su defecto una carpeta
python src/predict.py v2 ruta/a/carpeta/
```

Recibe una o varias fotos o carpetas, predice la clase de cada píxel y guarda una figura por foto en `results/predicciones/<experimento>/` con la foto original, la máscara predicha y la máscara sobre la foto, en caso de que la foto sea del dataset también muestra la máscara real, en la consola imprime las clases detectadas con el porcentaje de la foto que ocupa cada una, las figuras no se suben al repositorio porque pueden mostrar imágenes de ADE20K.

### Interfaz

```bash
streamlit run app/app.py
```

Abre en el navegador una página local en `http://localhost:8501` donde se elige el modelo, se sube una foto y se muestra la foto original, la máscara predicha, la máscara sobre la foto y las clases detectadas con el porcentaje de la foto que ocupa cada una y en caso de que la foto sea del dataset también muestra la máscara real con su exactitud y su mIoU.

### Pruebas del entorno

```bash
python src/data.py
python src/gpu_memory_probe.py
python src/gpu_benchmark.py
```

`data.py` revisa el pipeline de datos, `gpu_memory_probe.py` mide cuánta memoria de la GPU puede reservar TensorFlow y `gpu_benchmark.py` mide el tiempo por paso con distintos tamaños de batch, filtros y precisión numérica.

## Resultados

| Experimento | Cambio | mIoU validación | mIoU prueba |
|---|---|---|---|
| v1 | U-Net base desde cero | 0.144 | 0.144 |
| v2 | v1 con volteo horizontal aleatorio | 0.160 | 0.166 |
 
El detalle de cada experimento está en `results/experimentos/` y en el paper.

## Estado del proyecto

Esta es la versión final del proyecto, el modelo todavía se puede mejorar con los siguientes ajustes cada uno cambia una cosa respecto al anterior, los cuales son:

1. Dropout en el decodificador y regularización L2 para seguir reduciendo el sobreajuste (overfit).
2. Pesos por clase en la función de costo para mejorar las clases poco frecuentes.
3. Entrenar durante más épocas ya que la mejor época de v2 fue la última y tuvo posibilidad de mejorar.
4. Transfer learning con un codificador preentrenado en caso de que el modelo desde cero deje de funcionar.
5. Una interfaz más elaborada en html con diseño de tipo chat, esto no mejora el modelo pero si la experiencia de usuario.

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
