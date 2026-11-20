
"""
PRACTICA 3 - SESIONES 2, 3 Y 5
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

Compatible con Docker y Python 3.10.
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
# CONFIGURACION GENERAL
# ==========================================

BASE = Path(__file__).resolve().parent.parent
MODELO_ONNX = BASE / "models" / "modelo.onnx"

NUM_SENSORES = 24

# Estos limites deben coincidir con el dataset
# y el preprocesamiento utilizado al entrenar.
SENSOR_MIN = 0.0
SENSOR_MAX = 1.0

logger = logging.getLogger(__name__)


# ==========================================
# VALIDACION DE DATOS CON PYDANTIC
# ==========================================

class InputData(BaseModel):
    """Valida las lecturas de los 24 sensores."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    sensores: Annotated[
        list[float],
        Field(
            min_length=NUM_SENSORES,
            max_length=NUM_SENSORES,
            description="Lecturas de los 24 sensores.",
        ),
    ]

    @field_validator("sensores", mode="before")
    @classmethod
    def validar_tipos(cls, valor):
        """Rechaza valores no numericos y booleanos."""

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
    def validar_rangos(cls, sensores):
        """Verifica valores finitos y rangos."""

        for indice, valor in enumerate(
            sensores, start=1
        ):
            if not math.isfinite(valor):
                raise ValueError(
                    f"Sensor {indice}: "
                    "valor no finito."
                )

            if not (
                SENSOR_MIN <= valor <= SENSOR_MAX
            ):
                raise ValueError(
                    f"Sensor {indice}: {valor} "
                    f"fuera del rango "
                    f"[{SENSOR_MIN}, {SENSOR_MAX}]."
                )

        return sensores


# ==========================================
# CONVERSION SEGURA DE PREDICCIONES
# ==========================================

def convertir_prediccion(valor):
    """
    Convierte una salida ONNX a un tipo
    compatible con JSON.

    Admite cadenas de texto, enteros,
    flotantes y escalares de NumPy.
    """

    if isinstance(valor, np.generic):
        valor = valor.item()

    if isinstance(valor, bytes):
        valor = valor.decode("utf-8")

    if isinstance(valor, (str, int, bool)):
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
# CARGA DEL MODELO ONNX
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la API."""

    if not MODELO_ONNX.is_file():
        raise RuntimeError(
            "Modelo ONNX no encontrado: "
            f"{MODELO_ONNX}"
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
            "El modelo ONNX no tiene las "
            "24 variables de entrada esperadas."
        )

    if entrada.type != "tensor(float)":
        raise RuntimeError(
            "Tipo de entrada ONNX no esperado: "
            f"{entrada.type}. "
            "La API utiliza float32."
        )

    app.state.sesion_onnx = sesion
    app.state.nombre_entrada = entrada.name

    logger.info(
        "Modelo ONNX cargado correctamente."
    )
    logger.info(
        "Sensores esperados: %s",
        NUM_SENSORES,
    )

    try:
        yield
    finally:
        app.state.sesion_onnx = None
        app.state.nombre_entrada = None


# ==========================================
# CONFIGURACION DE FASTAPI
# ==========================================

app = FastAPI(
    title="API Machine Learning - Practica 3",
    description=(
        "API REST para predicciones del robot "
        "Wall-Following mediante Random Forest "
        "exportado a ONNX, con validacion Pydantic."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# ==========================================
# ENDPOINT GET /health
# ==========================================

@app.get("/health")
def health():
    """Comprueba que el modelo esta disponible."""

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
        "rango_permitido": {
            "minimo": SENSOR_MIN,
            "maximo": SENSOR_MAX,
        },
    }


# ==========================================
# ENDPOINT POST /predict
# ==========================================

@app.post("/predict")
def predict(datos: InputData):
    """
    Recibe 24 sensores, valida sus valores
    y ejecuta una prediccion con ONNX.
    """

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

    # Convertir los sensores a matriz (1, 24).
    entrada = np.asarray(
        [datos.sensores],
        dtype=np.float32,
    )

    # Verificar que los valores siguen siendo
    # finitos despues de convertir a float32.
    if not np.isfinite(entrada).all():
        raise HTTPException(
            status_code=422,
            detail=(
                "Valores no representables "
                "en float32."
            ),
        )

    try:
        # Ejecutar la inferencia ONNX.
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

        # Obtener la primera prediccion.
        salida = np.asarray(
            resultados[0]
        ).reshape(-1)

        if salida.size == 0:
            raise RuntimeError(
                "La salida ONNX esta vacia."
            )

        # CORRECCION PRINCIPAL:
        # No llamar .item() sobre un str.
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
        "cantidad_sensores": len(
            datos.sensores
        ),
        "validacion": "correcta",
    }
