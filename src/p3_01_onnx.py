
"""
PRACTICA 3 - SESION 1
Exportacion de Random Forest a ONNX.

Compara las predicciones del modelo original
con las obtenidas mediante ONNX Runtime.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


BASE = Path(__file__).resolve().parent.parent
MODELOS = BASE / "models"
REPORTES = BASE / "reports"
DATOS = BASE / "data" / "processed"

REPORTES.mkdir(parents=True, exist_ok=True)

RUTA_ORIGINAL = MODELOS / "random_forest.pkl"
RUTA_ONNX = MODELOS / "modelo.onnx"
RUTA_DATASET = DATOS / "dataset.parquet"


def cargar_modelo():
    """Carga el modelo Random Forest existente."""

    if not RUTA_ORIGINAL.exists():
        raise FileNotFoundError(
            f"No se encontro: {RUTA_ORIGINAL}"
        )

    modelo = joblib.load(RUTA_ORIGINAL)

    if not hasattr(modelo, "predict"):
        raise TypeError(
            "El archivo no contiene un predictor compatible."
        )

    if not hasattr(modelo, "n_features_in_"):
        raise ValueError(
            "No se pudo determinar el numero de variables."
        )

    return modelo


def exportar_onnx(modelo):
    """Convierte el estimador a ONNX."""

    cantidad_variables = int(modelo.n_features_in_)

    tipos_entrada = [
        (
            "input",
            FloatTensorType([None, cantidad_variables])
        )
    ]

    modelo_onnx = convert_sklearn(
        modelo,
        initial_types=tipos_entrada,
        target_opset=17
    )

    onnx.checker.check_model(modelo_onnx)

    RUTA_ONNX.write_bytes(
        modelo_onnx.SerializeToString()
    )

    return cantidad_variables


def preparar_datos(modelo):
    """
    Prepara muestras numericas para comparar
    ambos modelos usando la misma entrada.

    No modifica ni reentrena el estimador.
    """

    if not RUTA_DATASET.exists():
        raise FileNotFoundError(
            f"No se encontro: {RUTA_DATASET}"
        )

    df = pd.read_parquet(RUTA_DATASET)

    if hasattr(modelo, "feature_names_in_"):
        columnas = list(modelo.feature_names_in_)
    else:
        columnas = [
            f"sensor_{i}"
            for i in range(1, 25)
        ]

    faltantes = [
        col for col in columnas
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )

    if len(columnas) != modelo.n_features_in_:
        raise ValueError(
            "El numero de variables no coincide "
            "con el modelo entrenado."
        )

    if "conjunto" in df.columns:
        prueba = df[df["conjunto"] == "test"]
    else:
        prueba = df

    if prueba.empty:
        raise ValueError(
            "No existen observaciones para comparar."
        )

    X = prueba[columnas].head(100)

    if X.isna().any().any():
        raise ValueError(
            "Las muestras contienen valores faltantes."
        )

    return X.astype(np.float32)


def comparar_predicciones(modelo, X):
    """Compara las salidas de sklearn y ONNX."""

    sesion = ort.InferenceSession(
        str(RUTA_ONNX),
        providers=["CPUExecutionProvider"]
    )

    entrada = sesion.get_inputs()[0].name

    salidas = sesion.run(
        None,
        {entrada: X.to_numpy(dtype=np.float32)}
    )

    pred_original = modelo.predict(X)
    pred_onnx = np.asarray(salidas[0]).reshape(-1)

    coincidencias = int(
        np.sum(pred_original == pred_onnx)
    )

    total = len(pred_original)

    porcentaje = (
        100.0 * coincidencias / total
    )

    return {
        "muestras": total,
        "coincidencias": coincidencias,
        "diferencias": total - coincidencias,
        "coincidencia_porcentaje": porcentaje,
        "predicciones_original": (
            np.asarray(pred_original[:10]).tolist()
        ),
        "predicciones_onnx": (
            np.asarray(pred_onnx[:10]).tolist()
        )
    }


def main():
    """Ejecuta exportacion y validacion."""

    print("=== PRACTICA 3 - SESION 1 ===")

    modelo = cargar_modelo()

    variables = exportar_onnx(modelo)

    print(f"Variables de entrada: {variables}")
    print(f"Modelo ONNX: {RUTA_ONNX}")

    X = preparar_datos(modelo)

    resultado = comparar_predicciones(
        modelo, X
    )

    reporte = {
        "modelo_original": str(RUTA_ORIGINAL),
        "modelo_onnx": str(RUTA_ONNX),
        "variables": variables,
        **resultado
    }

    ruta_reporte = (
        REPORTES / "p3_01_comparacion_onnx.json"
    )

    ruta_reporte.write_text(
        json.dumps(
            reporte,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== COMPARACION ===")
    print(
        "Muestras:",
        resultado["muestras"]
    )
    print(
        "Coincidencia:",
        f"{resultado['coincidencia_porcentaje']:.2f}%"
    )
    print(
        "Diferencias:",
        resultado["diferencias"]
    )
    print("\nReporte:", ruta_reporte)

    if resultado["diferencias"] != 0:
        print(
            "\nATENCION: revisar las diferencias "
            "antes de desplegar la API."
        )
        raise SystemExit(1)

    print("\nSESION 1 COMPLETADA")


if __name__ == "__main__":
    main()
