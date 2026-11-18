
"""
PRACTICA 2 - SESION 9
Evaluacion del agente PPO frente a un agente aleatorio.

Entregables:
- reports/p2_09_evaluacion_rl.json
- reports/p2_09_evaluacion_rl.csv
- reports/p2_09_conclusiones.md
- reports/figures/p2_09_comparacion_rl.png
"""

from pathlib import Path
import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO


BASE = Path(__file__).resolve().parent.parent

MODELO = BASE / "models" / "ppo_cartpole.zip"
REPORTES = BASE / "reports"
FIGURAS = REPORTES / "figures"

REPORTES.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

EPISODIOS = 10
SEMILLA = 42


def evaluar_agente(modelo=None, aleatorio=False):
    """
    Evalua una politica en CartPole-v1.

    Usa las mismas semillas para facilitar
    la comparacion entre agentes.
    """

    entorno = gym.make("CartPole-v1")

    resultados = []

    for episodio in range(EPISODIOS):

        semilla = SEMILLA + episodio

        estado, info = entorno.reset(
            seed=semilla
        )

        entorno.action_space.seed(semilla)

        terminado = False
        recompensa_total = 0.0
        pasos = 0

        while not terminado:

            if aleatorio:
                accion = entorno.action_space.sample()
            else:
                accion, _ = modelo.predict(
                    estado,
                    deterministic=True
                )

            estado, recompensa, terminated, truncated, info = (
                entorno.step(int(accion))
            )

            recompensa_total += float(recompensa)
            pasos += 1

            terminado = terminated or truncated

        resultados.append({
            "episodio": episodio + 1,
            "recompensa": recompensa_total,
            "pasos": pasos
        })

    entorno.close()

    return resultados


def calcular_estadisticas(resultados):
    """Calcula estadisticas de las recompensas."""

    recompensas = [
        fila["recompensa"]
        for fila in resultados
    ]

    return {
        "recompensa_promedio": float(
            np.mean(recompensas)
        ),
        "recompensa_maxima": float(
            np.max(recompensas)
        ),
        "recompensa_minima": float(
            np.min(recompensas)
        ),
        "desviacion_estandar": float(
            np.std(recompensas)
        ),
        "episodios": len(recompensas)
    }


def generar_grafico(ppo, aleatorio):
    """Compara recompensas por episodio."""

    episodios = np.arange(1, EPISODIOS + 1)

    recompensas_ppo = [
        fila["recompensa"] for fila in ppo
    ]

    recompensas_aleatorio = [
        fila["recompensa"] for fila in aleatorio
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        episodios,
        recompensas_ppo,
        marker="o",
        linewidth=2,
        label="PPO entrenado"
    )

    plt.plot(
        episodios,
        recompensas_aleatorio,
        marker="s",
        linewidth=2,
        label="Agente aleatorio"
    )

    plt.title(
        "Sesion 9 - PPO vs Agente Aleatorio"
    )

    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")

    plt.xticks(episodios)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ruta = FIGURAS / "p2_09_comparacion_rl.png"

    plt.savefig(ruta, dpi=300)
    plt.close()

    print("\nGrafico guardado:", ruta)


def guardar_resultados(ppo, aleatorio):
    """Guarda resultados y conclusiones."""

    estadisticas_ppo = calcular_estadisticas(ppo)
    estadisticas_aleatorio = calcular_estadisticas(
        aleatorio
    )

    diferencia = (
        estadisticas_ppo["recompensa_promedio"]
        - estadisticas_aleatorio["recompensa_promedio"]
    )

    datos = {
        "sesion": 9,
        "entorno": "CartPole-v1",
        "episodios_por_agente": EPISODIOS,
        "PPO": estadisticas_ppo,
        "Aleatorio": estadisticas_aleatorio,
        "diferencia_promedio": float(diferencia)
    }

    ruta_json = REPORTES / "p2_09_evaluacion_rl.json"

    with open(
        ruta_json,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            datos,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    filas = []

    for nombre, resultados in [
        ("PPO", ppo),
        ("Aleatorio", aleatorio)
    ]:

        for fila in resultados:
            filas.append({
                "agente": nombre,
                **fila
            })

    tabla = pd.DataFrame(filas)

    ruta_csv = REPORTES / "p2_09_evaluacion_rl.csv"

    tabla.to_csv(
        ruta_csv,
        index=False
    )

    if diferencia > 0:
        interpretacion = (
            "PPO obtuvo una recompensa promedio superior "
            "a la del agente aleatorio."
        )
    elif diferencia < 0:
        interpretacion = (
            "PPO obtuvo una recompensa promedio inferior "
            "a la del agente aleatorio."
        )
    else:
        interpretacion = (
            "Ambos agentes obtuvieron la misma "
            "recompensa promedio."
        )

    conclusion = f"""# Practica 2 - Sesion 9

## Objetivo

Evaluar un agente PPO entrenado en CartPole-v1
y compararlo con un agente aleatorio.

## Metodologia

Se ejecutaron {EPISODIOS} episodios por agente
utilizando las mismas semillas iniciales.

PPO utiliza prediccion determinista.
El agente aleatorio selecciona acciones al azar.

## Resultados

| Metrica | PPO | Aleatorio |
|---|---:|---:|
| Promedio | {estadisticas_ppo['recompensa_promedio']:.2f} | {estadisticas_aleatorio['recompensa_promedio']:.2f} |
| Maxima | {estadisticas_ppo['recompensa_maxima']:.2f} | {estadisticas_aleatorio['recompensa_maxima']:.2f} |
| Minima | {estadisticas_ppo['recompensa_minima']:.2f} | {estadisticas_aleatorio['recompensa_minima']:.2f} |

Diferencia de recompensa promedio: {diferencia:.2f}

## Conclusion

{interpretacion}

Estos resultados corresponden a diez episodios
de simulacion. No garantizan el mismo rendimiento
con otras semillas o condiciones.
"""

    ruta_md = REPORTES / "p2_09_conclusiones.md"

    ruta_md.write_text(
        conclusion,
        encoding="utf-8"
    )

    print("\n=== RESULTADOS FINALES ===")

    print("\nPPO:")
    print(estadisticas_ppo)

    print("\nALEATORIO:")
    print(estadisticas_aleatorio)

    print(
        "\nDiferencia promedio:",
        round(diferencia, 2)
    )

    print("\nArchivos generados:")
    print(ruta_json)
    print(ruta_csv)
    print(ruta_md)


def main():
    """Ejecuta la evaluacion completa."""

    print("\n=== PRACTICA 2 - SESION 9 ===")

    if not MODELO.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo: {MODELO}"
        )

    print("\nCargando modelo PPO...")

    modelo = PPO.load(
        str(MODELO),
        device="cpu"
    )

    print("\nEvaluando agente PPO...")

    resultados_ppo = evaluar_agente(
        modelo=modelo
    )

    print("\nEvaluando agente aleatorio...")

    resultados_aleatorio = evaluar_agente(
        aleatorio=True
    )

    generar_grafico(
        resultados_ppo,
        resultados_aleatorio
    )

    guardar_resultados(
        resultados_ppo,
        resultados_aleatorio
    )

    print("\nSESION 9 COMPLETADA")


if __name__ == "__main__":
    main()
