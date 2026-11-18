
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


BASE = Path(__file__).resolve().parent.parent
MODELOS = BASE / "models"
DATOS = BASE / "data" / "processed"
REPORTES = BASE / "reports"

MODELO_ORIGINAL = MODELOS / "random_forest.pkl"
MODELO_ONNX = MODELOS / "modelo.onnx"

SENSORES = [f"sensor_{i}" for i in range(1, 25)]


def main():
    """Exporta Random Forest a ONNX y compara predicciones."""

    REPORTES.mkdir(parents=True, exist_ok=True)

    if not MODELO_ORIGINAL.is_file():
        raise FileNotFoundError(
            f"No se encontró: {MODELO_ORIGINAL}"
        )

    # Cargar modelo original de scikit-learn
    modelo = joblib.load(MODELO_ORIGINAL)

    if not hasattr(modelo, "predict"):
        raise TypeError(
            "El archivo no contiene un predictor compatible."
        )

    # Recuperar ejemplos reales del dataset
    df = pd.read_parquet(
        DATOS / "dataset.parquet"
    )

    X = df[SENSORES].head(100).astype(np.float32)

    if len(X) == 0:
        raise ValueError("El dataset no contiene registros.")

    # Convertir el modelo al formato ONNX
    tipo_entrada = [
        ("input", FloatTensorType([None, 24]))
    ]

    modelo_convertido = convert_sklearn(
        modelo,
        initial_types=tipo_entrada,
        options={id(modelo): {"zipmap": False}},
        target_opset=15
    )

    onnx.checker.check_model(modelo_convertido)

    # Guardar archivo ONNX
    MODELO_ONNX.write_bytes(
        modelo_convertido.SerializeToString()
    )

    # Cargar ONNX Runtime
    sesion = ort.InferenceSession(
        str(MODELO_ONNX),
        providers=["CPUExecutionProvider"]
    )

    nombre_entrada = sesion.get_inputs()[0].name

    # Predicciones del modelo original
    pred_original = np.asarray(
        modelo.predict(X)
    ).reshape(-1)

    # Predicciones del modelo ONNX
    salida = sesion.run(
        None,
        {nombre_entrada: X.to_numpy()}
    )

    pred_onnx = np.asarray(
        salida[0]
    ).reshape(-1)

    coincidencias = (
        pred_original.astype(str)
        == pred_onnx.astype(str)
    )

    porcentaje = float(
        coincidencias.mean() * 100
    )

    resumen = {
        "modelo_original": str(MODELO_ORIGINAL),
        "modelo_onnx": str(MODELO_ONNX),
        "muestras": len(X),
        "coincidencias": int(coincidencias.sum()),
        "porcentaje_coincidencia": porcentaje,
        "salida_original": pred_original[:10].tolist(),
        "salida_onnx": pred_onnx[:10].tolist()
    }

    ruta = REPORTES / "p3_01_comparacion.json"

    ruta.write_text(
        json.dumps(
            resumen,
            indent=4,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    print("\n=== PRACTICA 3 - SESION 1 ===")
    print("Modelo ONNX:", MODELO_ONNX)
    print("Muestras evaluadas:", len(X))
    print("Coincidencia:", f"{porcentaje:.2f}%")
    print("Reporte:", ruta)

    if not coincidencias.all():
        raise RuntimeError(
            "Las predicciones ONNX y originales "
            "no coinciden en todas las muestras."
        )

    print("\nSesion 1 completada correctamente.")


if __name__ == "__main__":
    main()
