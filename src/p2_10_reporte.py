
from pathlib import Path

import pandas as pd


# ============================================
# PRACTICA 2 - SESION 10
# DOCUMENTACION DE RESULTADOS
# ============================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "processed"
REPORTES = BASE / "reports"

REPORTES.mkdir(parents=True, exist_ok=True)

ARCHIVOS = {
    "cartpole": DATOS / "p2_cartpole_random.csv",
    "robot": DATOS / "p2_robot_env_resultados.csv",
    "ppo": DATOS / "p2_ppo_entrenamiento.csv",
    "comparacion": DATOS / "p2_09_comparacion.csv"
}


def cargar_csv(nombre, obligatorio=True):

    ruta = ARCHIVOS[nombre]

    if not ruta.exists():

        if obligatorio:
            raise FileNotFoundError(
                f"Falta el archivo: {ruta}"
            )

        return None

    return pd.read_csv(ruta)


def main():

    print("\n=== SESION 10: REPORTE ===")

    cartpole = cargar_csv("cartpole")
    robot = cargar_csv("robot")
    comparacion = cargar_csv("comparacion")

    ppo = cargar_csv(
        "ppo",
        obligatorio=False
    )

    # ----------------------------------------
    # SESION 6
    # ----------------------------------------

    recompensa_cartpole = (
        cartpole["Recompensa total"].mean()
    )

    # ----------------------------------------
    # SESION 7
    # ----------------------------------------

    resumen_robot = (
        robot.groupby("Politica")
        .agg(
            recompensa=("Recompensa total", "mean"),
            pasos=("Pasos", "mean"),
            exito=("Meta alcanzada", "mean")
        )
    )

    # ----------------------------------------
    # SESION 8
    # ----------------------------------------

    if ppo is not None and not ppo.empty:

        primeros = (
            ppo["Recompensa"]
            .head(20)
            .mean()
        )

        ultimos = (
            ppo["Recompensa"]
            .tail(20)
            .mean()
        )

        texto_ppo = f"""
Episodios registrados: {len(ppo)}.

Recompensa media de los primeros 20 episodios:
{primeros:.2f}.

Recompensa media de los últimos 20 episodios:
{ultimos:.2f}.

Estos valores describen la evolución observada
durante el entrenamiento. La evaluación del
agente se presenta en la Sesión 9.
"""
    else:

        texto_ppo = (
            "El historial CSV de entrenamiento PPO "
            "no esta disponible. El modelo se evalua "
            "en la Sesion 9."
        )

    # ----------------------------------------
    # SESION 9
    # ----------------------------------------

    resumen_9 = comparacion.set_index(
        "Politica"
    )

    recompensa_ppo = resumen_9.loc[
        "PPO", "recompensa_promedio"
    ]

    recompensa_reglas = resumen_9.loc[
        "Reglas", "recompensa_promedio"
    ]

    recompensa_aleatorio = resumen_9.loc[
        "Aleatorio", "recompensa_promedio"
    ]

    exito_ppo = resumen_9.loc[
        "PPO", "tasa_exito"
    ]

    if recompensa_ppo >= recompensa_reglas:

        conclusion = (
            "PPO alcanzo una recompensa media "
            "igual o superior a la politica de reglas."
        )

    elif recompensa_ppo > recompensa_aleatorio:

        conclusion = (
            "PPO supero la recompensa media del "
            "agente aleatorio, aunque no alcanzo "
            "el rendimiento de la politica de reglas."
        )

    else:

        conclusion = (
            "PPO no supero la recompensa media "
            "del agente aleatorio en esta evaluacion. "
            "Se recomienda revisar el entrenamiento."
        )

    # ----------------------------------------
    # GENERAR DOCUMENTO
    # ----------------------------------------

    reporte = f"""# Practica 2 - Reporte de aprendizaje por refuerzo

## Objetivo

Implementar y evaluar agentes de aprendizaje por
refuerzo utilizando Gymnasium y Stable Baselines3.

## Sesion 6 - CartPole

Se ejecuto un agente aleatorio durante cinco episodios.

Recompensa promedio: {recompensa_cartpole:.2f}.

El agente no fue entrenado y se utilizo como
referencia de comportamiento aleatorio en CartPole.

## Sesion 7 - Entorno personalizado

Se implemento RobotLineEnv, un entorno donde un
robot debe alcanzar una posicion objetivo.

Resultados:

{resumen_robot.to_string()}

## Sesion 8 - Entrenamiento PPO

Se entreno un agente PPO con 10000 pasos solicitados,
utilizando MlpPolicy en RobotLineEnv.

{texto_ppo}

## Sesion 9 - Evaluacion comparativa

Se evaluaron 30 episodios por politica.

{comparacion.to_string(index=False)}

Recompensa media PPO: {recompensa_ppo:.2f}.

Tasa de exito PPO: {exito_ppo:.2f}%.

## Conclusion

{conclusion}

Los resultados corresponden a una simulacion
academica. No representan pruebas en un robot fisico.

## Evidencias

- reports/figures/p2_cartpole_random.png
- reports/figures/p2_ppo_entrenamiento.png
- reports/figures/p2_09_recompensas.png
- reports/figures/p2_09_tasa_exito.png
- models/p2_ppo_robot.zip
"""

    ruta_reporte = (
        REPORTES / "p2_reporte_sesiones_6_9.md"
    )

    ruta_reporte.write_text(
        reporte,
        encoding="utf-8"
    )

    print("\n=== RESUMEN ===")
    print("CartPole promedio:", round(recompensa_cartpole, 2))
    print("PPO recompensa:", round(recompensa_ppo, 2))
    print("PPO tasa de exito:", round(exito_ppo, 2))

    print("\nReporte generado:")
    print(ruta_reporte)

    print("\nSesion 10 completada correctamente.")


if __name__ == "__main__":
    main()
