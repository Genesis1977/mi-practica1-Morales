
"""
PRACTICA 3 - SESION 3
Pruebas automaticas de validacion Pydantic.
"""

from pathlib import Path
import json

from fastapi.testclient import TestClient

from app.api import (
    app,
    NUM_SENSORES,
    SENSOR_MIN,
    SENSOR_MAX
)


BASE = Path(__file__).resolve().parent.parent
REPORTES = BASE / "reports"
REPORTES.mkdir(parents=True, exist_ok=True)


def main():
    """Ejecuta y documenta pruebas HTTP."""

    valor_valido = (SENSOR_MIN + SENSOR_MAX) / 2
    sensores = [valor_valido] * NUM_SENSORES

    casos = [
        {
            "nombre": "Entrada valida",
            "datos": {"sensores": sensores},
            "esperado": 200
        },
        {
            "nombre": "Solo tres sensores",
            "datos": {"sensores": [0.1, 0.2, 0.3]},
            "esperado": 422
        },
        {
            "nombre": "Sensor con texto",
            "datos": {
                "sensores": ["hola"] + sensores[1:]
            },
            "esperado": 422
        },
        {
            "nombre": "Sensor negativo fuera de rango",
            "datos": {
                "sensores": [SENSOR_MIN - 1] + sensores[1:]
            },
            "esperado": 422
        },
        {
            "nombre": "Sensor superior al maximo",
            "datos": {
                "sensores": [SENSOR_MAX + 1] + sensores[1:]
            },
            "esperado": 422
        },
        {
            "nombre": "Campo adicional",
            "datos": {
                "sensores": sensores,
                "usuario": "prueba"
            },
            "esperado": 422
        },
        {
            "nombre": "Sensor booleano",
            "datos": {
                "sensores": [True] + sensores[1:]
            },
            "esperado": 422
        },
        {
            "nombre": "Sensores ausentes",
            "datos": {},
            "esperado": 422
        }
    ]

    registros = []

    with TestClient(app) as cliente:

        salud = cliente.get("/health")
        assert salud.status_code == 200

        for caso in casos:
            respuesta = cliente.post(
                "/predict",
                json=caso["datos"]
            )

            correcto = (
                respuesta.status_code == caso["esperado"]
            )

            registros.append({
                "caso": caso["nombre"],
                "codigo_esperado": caso["esperado"],
                "codigo_obtenido": respuesta.status_code,
                "aprobado": correcto,
                "respuesta": respuesta.json()
            })

            estado = "OK" if correcto else "ERROR"

            print(
                f"[{estado}] {caso['nombre']} "
                f"HTTP {respuesta.status_code}"
            )

    ruta = REPORTES / "p3_03_pruebas.json"

    ruta.write_text(
        json.dumps(
            registros,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    aprobadas = sum(
        registro["aprobado"]
        for registro in registros
    )

    print(
        f"\nPruebas aprobadas: "
        f"{aprobadas}/{len(registros)}"
    )

    print("Reporte:", ruta)

    if aprobadas != len(registros):
        raise SystemExit(
            "Existen pruebas que requieren revision."
        )

    print("SESION 3: PRUEBAS APROBADAS")


if __name__ == "__main__":
    main()
