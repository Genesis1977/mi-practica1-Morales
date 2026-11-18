
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
import math
import traceback

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


BASE = Path(__file__).resolve().parent.parent
MODELO = BASE / "models" / "modelo.onnx"

sesion = None
entrada_modelo = None
tipo_entrada = None
cantidad_sensores = None


class InputData(BaseModel):
    """Valida las 24 lecturas de los sensores."""

    sensores: Annotated[
        list[float],
        Field(min_length=24, max_length=24)
    ]

    @field_validator("sensores", mode="before")
    @classmethod
    def validar_numeros(cls, valor):
        if not isinstance(valor, list):
            raise ValueError("Debe enviar una lista.")

        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            for x in valor
        ):
            raise ValueError(
                "Los sensores deben contener números."
            )

        return valor

    @field_validator("sensores")
    @classmethod
    def validar_rangos(cls, valores):
        if any(not math.isfinite(x) for x in valores):
            raise ValueError(
                "No se permiten valores infinitos o NaN."
            )

        if any(x < 0 for x in valores):
            raise ValueError(
                "Los sensores no pueden ser negativos."
            )

        return valores


class PredictionResponse(BaseModel):
    prediccion: int | float | str
    modelo: str
    cantidad_sensores: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo ONNX una sola vez."""

    global sesion
    global entrada_modelo
    global tipo_entrada
    global cantidad_sensores

    if not MODELO.exists():
        raise RuntimeError(
            f"No existe el modelo: {MODELO}. "
            "Ejecuta primero p3_01_onnx.py."
        )

    sesion = ort.InferenceSession(
        str(MODELO),
        providers=["CPUExecutionProvider"]
    )

    entradas = sesion.get_inputs()
    salidas = sesion.get_outputs()

    print("\n=== INFORMACION DEL MODELO ONNX ===")

    for entrada in entradas:
        print(
            "Entrada:",
            entrada.name,
            entrada.type,
            entrada.shape
        )

    for salida in salidas:
        print(
            "Salida:",
            salida.name,
            salida.type,
            salida.shape
        )

    if len(entradas) != 1:
        raise RuntimeError(
            "La API espera un modelo con una entrada."
        )

    entrada_modelo = entradas[0].name
    tipo_entrada = entradas[0].type

    forma = entradas[0].shape

    if len(forma) != 2:
        raise RuntimeError(
            "Se esperaba una entrada de dos dimensiones."
        )

    cantidad_sensores = forma[1]

    if cantidad_sensores != 24:
        raise RuntimeError(
            "El modelo ONNX no espera 24 sensores. "
            f"Forma encontrada: {forma}"
        )

    print("Modelo cargado correctamente.")

    yield

    sesion = None
    entrada_modelo = None


app = FastAPI(
    title="API Machine Learning - Practica 3",
    description="Predicciones Wall-Following mediante ONNX.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    """Verifica el estado de la API."""

    return {
        "estado": "ok",
        "modelo_cargado": sesion is not None,
        "modelo": "Random Forest ONNX"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(datos: InputData):
    """Predice a partir de 24 sensores validados."""

    if sesion is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible."
        )

    if tipo_entrada == "tensor(float)":
        dtype = np.float32
    elif tipo_entrada == "tensor(double)":
        dtype = np.float64
    else:
        raise HTTPException(
            status_code=500,
            detail="Tipo de entrada ONNX no compatible."
        )

    valores = np.asarray(
        [datos.sensores],
        dtype=dtype
    )

    try:
        salidas = sesion.run(
            None,
            {entrada_modelo: valores}
        )

        if not salidas:
            raise RuntimeError(
                "ONNX no devolvió resultados."
            )

        # La primera salida del clasificador suele
        # contener la etiqueta predicha.
        primera_salida = salidas[0]

        prediccion = np.asarray(
            primera_salida
        ).reshape(-1)[0]

        if isinstance(prediccion, np.generic):
            prediccion = prediccion.item()

        if isinstance(prediccion, bytes):
            prediccion = prediccion.decode("utf-8")

        if not isinstance(
            prediccion, (int, float, str)
        ):
            raise TypeError(
                "La salida del modelo no es una "
                "etiqueta de clasificación válida."
            )

        return PredictionResponse(
            prediccion=prediccion,
            modelo="Random Forest ONNX",
            cantidad_sensores=len(datos.sensores)
        )

    except Exception as error:
        print("\n=== ERROR DE INFERENCIA ===")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Error durante la inferencia."
        ) from error
