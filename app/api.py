
"""
PRACTICA 3 - SESIONES 2, 3, 5 Y 6

API REST para el robot Wall-Following.
FastAPI + Pydantic + ONNX Runtime.

Endpoints:
    GET  /health
    POST /predict

Validaciones:
    - Exactamente 24 sensores.
    - Solo numeros reales (no booleanos ni cadenas).
    - Valores finitos.
    - Valores representables en float32.
    - Sin campos adicionales.
    - Limites configurables.

IMPORTANTE:
La API recibe sensores ya transformados en la
misma escala y orden usados para entrenar el modelo.

Compatible con Python 3.10 y Docker.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import logging
import math

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ==========================================
# 1. CONFIGURACION
# ==========================================

BASE = Path(__file__).resolve().parent.parent
MODELO_ONNX = BASE / "models" / "modelo.onnx"

NUM_SENSORES = 24

# El dataset procesado contiene valores negativos
# y superiores a 1.
#
# No se aplican limites estadisticos hasta
# definir rangos justificados para cada sensor.
#
# La validacion de float32 permanece activa.
SENSOR_MIN = None
SENSOR_MAX = None

FLOAT32_MAX = float(np.finfo(np.float32).max)

logger = logging.getLogger(__name__)


# ==========================================
# 2. VALIDACION DE ENTRADA
# ==========================================

class InputData(BaseModel):
    """Esquema de entrada para 24 sensores."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    sensores: Annotated[
        list[float],
        Field(
            min_length=NUM_SENSORES,
            max_length=NUM_SENSORES,
            description=(
                "24 valores numericos finitos, "
                "en la escala del modelo."
            ),
        ),
    ]

    @field_validator("sensores", mode="before")
    @classmethod
    def validar_tipos(cls, valor):
        """Rechaza entradas que no sean numeros JSON."""

        if not isinstance(valor, list):
            raise ValueError(
                "sensores debe ser una lista."
            )

        for indice, sensor in enumerate(
            valor, start=1
        ):
            if isinstance(sensor, bool):
                raise ValueError(
                    f"Sensor {indice}: "
                    "no se permiten booleanos."
                )

            if not isinstance(sensor, (int, float)):
                raise ValueError(
                    f"Sensor {indice}: "
                    "debe ser numerico."
                )

        return valor

    @field_validator("sensores")
    @classmethod
    def validar_valores(cls, sensores):
        """Valida finitud, float32 y limites opcionales."""

        for indice, valor in enumerate(
            sensores, start=1
        ):
            if not math.isfinite(valor):
                raise ValueError(
                    f"Sensor {indice}: valor no finito."
                )

            if abs(valor) > FLOAT32_MAX:
                raise ValueError(
                    f"Sensor {indice}: fuera del "
                    "rango representable en float32."
                )

            if (
                SENSOR_MIN is not None
                and valor < SENSOR_MIN
            ):
                raise ValueError(
                    f"Sensor {indice}: menor "
                    f"que {SENSOR_MIN}."
                )

            if (
                SENSOR_MAX is not None
                and valor > SENSOR_MAX
            ):
                raise ValueError(
                    f"Sensor {indice}: mayor "
                    f"que {SENSOR_MAX}."
                )

        return sensores


# ==========================================
# 3. CONVERSION DE SALIDA ONNX
# ==========================================

def convertir_prediccion(valor):
    """
    Convierte un escalar ONNX/NumPy
    a un tipo compatible con JSON.
    """

    if isinstance(valor, np.generic):
        valor = valor.item()

    if isinstance(valor, bytes):
        valor = valor.decode("utf-8")

    if isinstance(valor, (str, bool, int)):
        return valor

    if isinstance(valor, float):
        if not math.isfinite(valor):
            raise ValueError(
                "La prediccion no es finita."
            )
        return valor

    raise TypeError(
        "Tipo de prediccion no soportado: "
        f"{type(valor).__name__}"
    )


# ==========================================
# 4. CARGA DEL MODELO
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga ONNX una sola vez al iniciar."""

    if not MODELO_ONNX.is_file():
        raise RuntimeError(
            f"Modelo ONNX no encontrado: {MODELO_ONNX}"
        )

    sesion = ort.InferenceSession(
        str(MODELO_ONNX),
        providers=["CPUExecutionProvider"],
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
            "El modelo ONNX debe recibir "
            "24 variables de entrada."
        )

    if entrada.type != "tensor(float)":
        raise RuntimeError(
            "El modelo ONNX no acepta float32. "
            f"Tipo detectado: {entrada.type}"
        )

    app.state.sesion_onnx = sesion
    app.state.nombre_entrada = entrada.name

    logger.info(
        "Modelo ONNX cargado correctamente."
    )
    logger.info(
        "Sensores esperados: %s",
        NUM_SENSORES
    )

    try:
        yield
    finally:
        app.state.sesion_onnx = None
        app.state.nombre_entrada = None


# ==========================================
# 5. FASTAPI
# ==========================================

app = FastAPI(
    title="API Machine Learning - Practica 3",
    description=(
        "API REST del robot Wall-Following. "
        "Predicciones con Random Forest ONNX "
        "y validacion mediante Pydantic."
    ),
    version="1.3.1",
    lifespan=lifespan,
)


# ==========================================
# 6. GET /health
# ==========================================

@app.get("/health")
def health():
    """Comprueba la disponibilidad del modelo."""

    sesion = getattr(
        app.state,
        "sesion_onnx",
        None,
    )

    if sesion is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible.",
        )

    return {
        "estado": "ok",
        "modelo": "Random Forest ONNX",
        "sensores_esperados": NUM_SENSORES,
        "validacion": {
            "tipos": "numericos",
            "cantidad": NUM_SENSORES,
            "valores_finitos": True,
            "compatible_float32": True,
            "limites_sensor": {
                "minimo": SENSOR_MIN,
                "maximo": SENSOR_MAX,
            },
        },
        "entrada": "sensores transformados",
    }


# ==========================================
# 7. POST /predict
# ==========================================

@app.post("/predict")
def predict(datos: InputData):
    """Valida sensores y ejecuta la inferencia ONNX."""

    sesion = getattr(
        app.state,
        "sesion_onnx",
        None,
    )

    if sesion is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible.",
        )

    entrada = np.asarray(
        [datos.sensores],
        dtype=np.float32,
    )

    if entrada.shape != (1, NUM_SENSORES):
        raise HTTPException(
            status_code=422,
            detail="Dimensiones de entrada invalidas.",
        )

    if not np.isfinite(entrada).all():
        raise HTTPException(
            status_code=422,
            detail=(
                "Valores no representables "
                "en float32."
            ),
        )

    try:
        resultados = sesion.run(
            None,
            {
                app.state.nombre_entrada: entrada
            },
        )

        if not resultados:
            raise RuntimeError(
                "ONNX no devolvio resultados."
            )

        salida = np.asarray(
            resultados[0]
        ).reshape(-1)

        if salida.size == 0:
            raise RuntimeError(
                "La salida ONNX esta vacia."
            )

        prediccion = convertir_prediccion(
            salida[0]
        )

    except Exception as error:
        logger.exception(
            "Error durante la inferencia ONNX."
        )

        raise HTTPException(
            status_code=500,
            detail="Error interno de inferencia.",
        ) from error

    return {
        "prediccion": prediccion,
        "modelo": "Random Forest ONNX",
        "cantidad_sensores": len(datos.sensores),
        "validacion": "correcta",
    }
