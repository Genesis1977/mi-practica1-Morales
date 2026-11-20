
"""
PRACTICA 3 - SESION 6
Profiling de RAM y latencia.

Entregables:
- reports/profiling.json
- reports/profiling.png

Mediciones:
1. RAM del contenedor Docker.
2. Latencia de 100 solicitudes HTTP.
3. Latencia de 100 inferencias ONNX.
4. Latencia de 100 inferencias Random Forest.
5. Comparacion de predicciones.
"""

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import requests


BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "processed" / "dataset.parquet"
MODELO_ORIGINAL = BASE / "models" / "random_forest.pkl"
MODELO_ONNX = BASE / "models" / "modelo.onnx"
REPORTES = BASE / "reports"

API_URL = "http://localhost:8000"
CONTENEDOR = "practica3-api"
NUM_SENSORES = 24
NUM_PRUEBAS = 100

COLUMNAS = [
    f"sensor_{i}" for i in range(1, NUM_SENSORES + 1)
]


def convertir_json(valor):
    """Convierte escalares NumPy a tipos JSON."""
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def resumen_tiempos(tiempos):
    """Calcula estadisticas de latencia en milisegundos."""
    datos = np.asarray(tiempos, dtype=float)

    return {
        "peticiones": len(datos),
        "promedio_ms": round(float(np.mean(datos)), 4),
        "mediana_ms": round(float(np.median(datos)), 4),
        "minimo_ms": round(float(np.min(datos)), 4),
        "maximo_ms": round(float(np.max(datos)), 4),
        "p95_ms": round(float(np.percentile(datos, 95)), 4),
        "desviacion_ms": round(float(np.std(datos)), 4),
    }


def convertir_memoria_mb(texto):
    """Convierte una medida de Docker a MB decimales."""
    coincidencia = re.match(
        r"^\s*([\d.,]+)\s*([KMGT]?i?B)\s*$",
        texto,
        re.IGNORECASE,
    )

    if not coincidencia:
        raise ValueError(
            f"Unidad de memoria no reconocida: {texto}"
        )

    cantidad = float(coincidencia.group(1).replace(",", "."))
    unidad = coincidencia.group(2).lower()

    factores = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }

    return round(cantidad * factores[unidad] / 1e6, 3)


def medir_ram_docker():
    """Obtiene la memoria usada por el contenedor."""
    comando = [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{.MemUsage}}",
        CONTENEDOR,
    ]

    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    linea = resultado.stdout.strip()

    if not linea:
        raise RuntimeError(
            "Docker no devolvio datos de memoria."
        )

    memoria_usada = linea.split("/")[0].strip()

    return {
        "contenedor": CONTENEDOR,
        "memoria_usada": memoria_usada,
        "ram_mb": convertir_memoria_mb(memoria_usada),
    }


def cargar_muestra():
    """Obtiene una fila real con 24 sensores finitos y representables."""
    if not DATASET.is_file():
        raise FileNotFoundError(f"No existe el dataset: {DATASET}")

    df = pd.read_parquet(DATASET)
    faltantes = [col for col in COLUMNAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas de sensores: {faltantes}")

    sensores = df[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    matriz = sensores.to_numpy(dtype=np.float64)
    filas_validas = np.isfinite(matriz).all(axis=1)
    filas_validas &= (np.abs(matriz) <= np.finfo(np.float32).max).all(axis=1)

    if not filas_validas.any():
        raise ValueError("No hay filas con 24 sensores finitos representables en float32.")

    muestra = matriz[np.flatnonzero(filas_validas)[0]].astype(np.float32).reshape(1, NUM_SENSORES)
    return muestra


def medir_api(muestra):
    """Mide 100 solicitudes HTTP al endpoint /predict."""
    tiempos = []
    predicciones = []

    datos = {
        "sensores": muestra.reshape(-1).astype(float).tolist()
    }

    with requests.Session() as cliente:
        # Calentamiento: no se incluye en las 100 mediciones.
        respuesta = cliente.post(
            f"{API_URL}/predict",
            json=datos,
            timeout=15,
        )
        if not respuesta.ok:
            raise RuntimeError(
                f"La API rechazo la muestra real (HTTP {respuesta.status_code}): "
                f"{respuesta.text[:700]}. Verifica que app/api.py acepte "
                "sensores transformados y que hayas reconstruido Docker."
            )

        for numero in range(NUM_PRUEBAS):
            inicio = time.perf_counter()

            respuesta = cliente.post(
                f"{API_URL}/predict",
                json=datos,
                timeout=15,
            )

            respuesta.raise_for_status()
            contenido = respuesta.json()

            duracion = (
                time.perf_counter() - inicio
            ) * 1000

            tiempos.append(duracion)
            predicciones.append(contenido["prediccion"])

            if (numero + 1) % 20 == 0:
                print(
                    f"API: {numero + 1}/{NUM_PRUEBAS}"
                )

    if len(set(map(str, predicciones))) != 1:
        raise RuntimeError(
            "La API devolvio resultados inconsistentes "
            "para la misma entrada."
        )

    return resumen_tiempos(tiempos), predicciones[0]


def medir_modelo(modelo, muestra):
    """Mide 100 inferencias originales conservando nombres de columnas."""
    tiempos = []
    muestra_df = pd.DataFrame(muestra, columns=COLUMNAS)
    modelo.predict(muestra_df)  # Calentamiento

    for _ in range(NUM_PRUEBAS):
        inicio = time.perf_counter()
        modelo.predict(muestra_df)
        tiempos.append((time.perf_counter() - inicio) * 1000)

    prediccion = convertir_json(np.asarray(modelo.predict(muestra_df)).reshape(-1)[0])
    return resumen_tiempos(tiempos), prediccion


def medir_onnx(sesion, muestra):
    """Mide 100 inferencias directas con ONNX Runtime."""
    nombre_entrada = sesion.get_inputs()[0].name
    entrada = {nombre_entrada: muestra}

    sesion.run(None, entrada)

    tiempos = []

    for _ in range(NUM_PRUEBAS):
        inicio = time.perf_counter()
        sesion.run(None, entrada)
        tiempos.append(
            (time.perf_counter() - inicio) * 1000
        )

    salida = sesion.run(None, entrada)
    prediccion = convertir_json(
        np.asarray(salida[0]).reshape(-1)[0]
    )

    return resumen_tiempos(tiempos), prediccion


def crear_grafico(api, onnx, original):
    """Genera el grafico comparativo de latencia."""
    nombres = [
        "API Docker",
        "ONNX directo",
        "Random Forest",
    ]

    promedios = [
        api["promedio_ms"],
        onnx["promedio_ms"],
        original["promedio_ms"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(nombres, promedios)

    ax.set_title("Practica 3 - Profiling de latencia")
    ax.set_ylabel("Latencia promedio (ms)")
    ax.set_ylim(0, max(promedios) * 1.2 if max(promedios) else 1)

    for barra, valor in zip(barras, promedios):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            f"{valor:.3f} ms",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    destino = REPORTES / "profiling.png"
    fig.savefig(destino, dpi=160)
    plt.close(fig)

    return destino


def main():
    """Ejecuta las mediciones y guarda los entregables."""
    REPORTES.mkdir(parents=True, exist_ok=True)

    print("\n=== PRACTICA 3 - SESION 6 ===")

    print("\n[1] Comprobando API...")
    respuesta = requests.get(
        f"{API_URL}/health",
        timeout=15,
    )
    respuesta.raise_for_status()

    estado_api = respuesta.json()
    print("API:", estado_api["estado"])
    if estado_api.get("sensores_esperados") != NUM_SENSORES:
        raise RuntimeError("La API no espera 24 sensores.")

    print("\n[2] Cargando muestra real...")
    muestra = cargar_muestra()
    print("Forma:", muestra.shape)
    print("Rango de la muestra:", float(muestra.min()), "a", float(muestra.max()))

    print("\n[3] Cargando modelos...")
    modelo_original = joblib.load(MODELO_ORIGINAL)
    sesion_onnx = ort.InferenceSession(
        str(MODELO_ONNX),
        providers=["CPUExecutionProvider"],
    )

    print("\n[4] Midiendo RAM inicial...")
    ram_antes = medir_ram_docker()
    print("RAM inicial:", ram_antes["ram_mb"], "MB")

    print("\n[5] Midiendo 100 peticiones HTTP...")
    api, pred_api = medir_api(muestra)

    print("\n[6] Midiendo 100 inferencias ONNX...")
    onnx, pred_onnx = medir_onnx(sesion_onnx, muestra)

    print("\n[7] Midiendo 100 inferencias originales...")
    original, pred_original = medir_modelo(
        modelo_original, muestra
    )

    print("\n[8] Midiendo RAM final...")
    ram_despues = medir_ram_docker()
    print("RAM final:", ram_despues["ram_mb"], "MB")

    coincidencia = (
        str(pred_api) == str(pred_onnx) == str(pred_original)
    )

    reporte = {
        "practica": "P3",
        "sesion": 6,
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": str(DATASET.relative_to(BASE)),
        "numero_sensores": NUM_SENSORES,
        "numero_pruebas": NUM_PRUEBAS,
        "rango_muestra": {
            "minimo": float(muestra.min()),
            "maximo": float(muestra.max()),
        },
        "unidad_latencia": "milisegundos",
        "ram_contenedor": {
            "antes": ram_antes,
            "despues": ram_despues,
            "metodo": "docker stats --no-stream",
        },
        "latencia": {
            "api_docker": api,
            "onnx_directo": onnx,
            "random_forest_original": original,
        },
        "predicciones": {
            "api": pred_api,
            "onnx": pred_onnx,
            "original": pred_original,
            "coinciden_en_muestra": coincidencia,
        },
        "metodologia": {
            "entrada": "Una muestra transformada real reutilizada en las 100 pruebas",
            "calentamiento": "Una ejecucion por metodo",
            "api": "HTTP local hacia Docker",
            "modelos": "Inferencia local en Windows",
            "nota": (
                "RAM medida solo antes y despues, no representa el pico. "
                "La latencia HTTP incluye transporte "
                "y procesamiento. La comparacion local "
                "de modelos no aisla diferencias de "
                "entorno ni garantiza equivalencia "
                "del preprocesamiento."
            ),
        },
        "equipo": {
            "cpu_logicos": psutil.cpu_count(),
            "ram_total_gb": round(
                psutil.virtual_memory().total / 1e9, 2
            ),
        },
    }

    destino_json = REPORTES / "profiling.json"

    with destino_json.open(
        "w", encoding="utf-8"
    ) as archivo:
        json.dump(
            reporte,
            archivo,
            indent=4,
            ensure_ascii=False,
            default=convertir_json,
        )

    destino_grafico = crear_grafico(
        api, onnx, original
    )

    print("\n=== RESULTADOS ===")
    print("API promedio:", api["promedio_ms"], "ms")
    print("ONNX promedio:", onnx["promedio_ms"], "ms")
    print(
        "Original promedio:",
        original["promedio_ms"], "ms"
    )
    print("Predicciones coinciden:", coincidencia)
    print("JSON:", destino_json)
    print("Grafico:", destino_grafico)

    if not coincidencia:
        print(
            "\nADVERTENCIA: las predicciones difieren. "
            "Revisa clases y preprocesamiento."
        )


if __name__ == "__main__":
    main()
