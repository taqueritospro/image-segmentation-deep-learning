''' Mide cuánta memoria de la GPU puede reservar TensorFlow con distintas configuraciones de memoria '''

import json
import os
import subprocess
import sys

# Cada configuración se prueba en un proceso nuevo, porque la memoria de la GPU solo se puede configurar al iniciar TensorFlow
MODES = {
    "crecimiento": {"memory_limit": None, "allocator": None},
    "limite_3500": {"memory_limit": 3500, "allocator": None},
    "async": {"memory_limit": None, "allocator": "cuda_malloc_async"},
}

# Tamaño de cada bloque que se reserva y número máximo de bloques, que equivale a 6.4 GB
BLOCK_MB = 64
MAX_BLOCKS = 100


def probe(memory_limit):
    ''' Reserva bloques de memoria en la GPU hasta que ya no se puede y regresa cuántos MB se lograron '''

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return {"estado": "sin GPU", "mb_reservados": 0}

    if memory_limit:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
    elif os.environ.get("TF_GPU_ALLOCATOR") != "cuda_malloc_async":
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # Los bloques se guardan en una lista para que no se liberen mientras se siguen reservando
    blocks = []
    floats_per_block = BLOCK_MB * 1024 * 1024 // 4
    try:
        with tf.device("GPU:0"):
            for _ in range(MAX_BLOCKS):
                blocks.append(tf.ones((floats_per_block,), dtype=tf.float32))
    except Exception:
        pass

    return {"estado": "ok", "mb_reservados": len(blocks) * BLOCK_MB}


def run_mode(name, mode):
    ''' Lanza la prueba de una configuración en un proceso separado y lee su resultado '''

    env = os.environ.copy()

    # Se ocultan los mensajes informativos de TensorFlow para que la salida sea legible
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if mode["allocator"]:
        env["TF_GPU_ALLOCATOR"] = mode["allocator"]

    command = [sys.executable, __file__, "--child", str(mode["memory_limit"] or 0)]
    process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)

    for line in process.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])

    # Si el proceso no imprimió resultado se muestra el final de su error
    error_lines = process.stderr.strip().splitlines()
    return {"estado": "error: " + (error_lines[-1] if error_lines else f"código {process.returncode}"), "mb_reservados": 0}


def main():
    ''' Prueba las tres configuraciones de memoria e imprime cuánta memoria logró reservar cada una '''

    # Modo interno: el proceso hijo hace la prueba e imprime su resultado
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        memory_limit = int(sys.argv[2]) or None
        print("RESULT " + json.dumps(probe(memory_limit)), flush=True)
        return

    for name, mode in MODES.items():
        print(f"Probando {name}...", flush=True)
        result = run_mode(name, mode)
        print(f"  {result['estado']}, {result['mb_reservados']} MB reservados", flush=True)


if __name__ == "__main__":
    main()
