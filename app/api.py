
"""
PRACTICA 3 - SESION 3
API REST con FastAPI, ONNX y validacion Pydantic.

Endpoints:
GET  /health
POST /predict

Validaciones:
- Exactamente 24 sensores.
- Solo numeros reales, sin booleanos ni cadenas.
- Valores finitos.
- Rango permitido configurable.
- Sin campos adicionales.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import math

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)


BASE = Path(__file__).resolve().parent.parent
MODELO_ONNX = BASE / "models" / "modelo.onnx"

# AJUSTAR SOLO DESPUES DE REVISAR EL DATASET
SENSOR_MIN = 0.0
SENSOR_MAX = 1.0
NUM_SENSORES = 24


class InputData(BaseModel):
    """Esquema estricto para las lecturas de sensores."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    sensores: Annotated[
        list[float],
        Field(
            min_length=NUM_SENSORES,
            max_length=NUM_SENSORES,
            description="Lecturas de los 24 sensores"
        )
    ]

    @field_validator("sensores", mode="before")
    @classmethod
    def validar_tipos(cls, valor):
        """Rechaza cadenas, booleanos y otros tipos."""

        if not isinstance(valor, list):
            raise ValueError(
                "sensores debe ser una lista."
            )

        for indice, sensor in enumerate(valor, start=1):

            if isinstance(sensor, bool):
                raise ValueError(
                    f"Sensor {indice}: no se permiten booleanos."
                )

            if not isinstance(sensor, (int, float)):
                raise ValueError(
                    f"Sensor {indice}: debe ser numerico."
                )

        return valor

    @field_validator("sensores")
    @classmethod
    def validar_rangos(cls, sensores):
        """Comprueba finitud y rango de cada sensor."""

        for indice, valor in enumerate(
            sensores, start=1
        ):

            if not math.isfinite(valor):
                raise ValueError(
                    f"Sensor {indice}: valor no finito."
                )

            if not SENSOR_MIN <= valor <= SENSOR_MAX:
                raise ValueError(
                    f"Sensor {indice}: {valor} fuera "
                    f"del rango [{SENSOR_MIN}, {SENSOR_MAX}]."
                )

        return sensores


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga y libera la sesion ONNX."""

    if not MODELO_ONNX.is_file():
        raise RuntimeError(
            f"Modelo ONNX no encontrado: {MODELO_ONNX}"
        )

    sesion = ort.InferenceSession(
        str(MODELO_ONNX),
        providers=["CPUExecutionProvider"]
    )

    entradas = sesion.get_inputs()

    if len(entradas) != 1:
        raise RuntimeError(
            "Se esperaba una sola entrada ONNX."
        )

    entrada = entradas[0]

    if (
        len(entrada.shape) != 2
        or entrada.shape[1] != NUM_SENSORES
    ):
        raise RuntimeError(
            "El modelo ONNX no tiene las "
            "24 variables de entrada esperadas."
        )

    app.state.sesion_onnx = sesion
    app.state.nombre_entrada = entrada.name

    print("Modelo ONNX cargado correctamente.")
    print("Sensores esperados:", NUM_SENSORES)

    yield

    app.state.sesion_onnx = None


app = FastAPI(
    title="API Machine Learning - Practica 3",
    description=(
        "API de prediccion Random Forest "
        "con ONNX y validacion Pydantic."
    ),
    version="1.1.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    """Devuelve el estado de la API."""

    if getattr(app.state, "sesion_onnx", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )

    return {
        "estado": "ok",
        "modelo": "Random Forest ONNX",
        "sensores_esperados": NUM_SENSORES,
        "rango_permitido": {
            "minimo": SENSOR_MIN,
            "maximo": SENSOR_MAX
        }
    }


@app.post("/predict")
def predict(datos: InputData):
    """Valida las lecturas y ejecuta la inferencia."""

    sesion = getattr(
        app.state, "sesion_onnx", None
    )

    if sesion is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )

    entrada = np.asarray(
        [datos.sensores],
        dtype=np.float32
    )

    if not np.isfinite(entrada).all():
        raise HTTPException(
            status_code=422,
            detail="Valores no representables en float32."
        )

    try:
        resultados = sesion.run(
            None,
            {
                app.state.nombre_entrada: entrada
            }
        )

        prediccion = np.asarray(
            resultados[0]
        ).reshape(-1)[0].item()

    except Exception as error:
        print("Error ONNX:", error)

        raise HTTPException(
            status_code=500,
            detail="Error interno de inferencia."
        ) from error

    return {
        "prediccion": prediccion,
        "modelo": "Random Forest ONNX",
        "cantidad_sensores": len(datos.sensores),
        "validacion": "correcta"
    }
